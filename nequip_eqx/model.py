import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


class Linear(eqx.Module):
    weights: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = jax.random.normal(key, (in_size, out_size)) * scale
        self.bias = jnp.zeros(out_size) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(eqx.Module):
    layers: list[Linear]
    activation: Callable

    def __init__(
        self,
        sizes,
        activation=jax.nn.silu,
        *,
        init_scale: float = 1.0,
        use_bias: bool = False,
        key: jax.Array,
    ):
        self.activation = activation

        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NequipConvolution(eqx.Module):
    output_irreps: e3nn.Irreps = eqx.field(static=True)

    avg_n_neighbors: float
    radial_mlp: MLP
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear
    skip: e3nn.equinox.Linear

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors

        tp_irreps = e3nn.tensor_product(
            input_irreps, sh_irreps, filter_ir_out=output_irreps
        )

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            key=k1,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=output_irreps,
            key=k3,
        )

        # skip connection has per-species weights
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            key=k4,
        )

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        messages = self.linear_1(features)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(jax.lax.stop_gradient(self.avg_n_neighbors))

        features = self.linear_2(messages_agg) + self.skip(species, features)

        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )


class Nequip(eqx.Module):
    lmax: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)

    shift: float
    scale: float
    atom_energies: jax.Array
    layers: list[NequipConvolution]
    readout: e3nn.equinox.Linear

    def __init__(
        self,
        key,
        n_species,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_size: int = 128,
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
    ):
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atom_energies = (
            jnp.array(atom_energies)
            if atom_energies is not None
            else jnp.zeros(n_species, dtype=jnp.float32)
        )
        input_irreps = e3nn.Irreps(f"{n_species}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = hidden_size * e3nn.s2_irreps(lmax)
        self.layers = []

        key, *subkeys = jax.random.split(key, n_layers + 1)
        for i in range(n_layers):
            self.layers.append(
                NequipConvolution(
                    key=subkeys[i],
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    n_species=n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_n_neighbors=avg_n_neighbors,
                )
            )

        self.readout = e3nn.equinox.Linear(
            irreps_in=hidden_irreps, irreps_out="0e", key=key
        )

    def node_energies(self, positions: jax.Array, data: jraph.GraphsTuple):
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"),
            jax.nn.one_hot(data.nodes["species"], self.n_species),
        )

        r = positions[data.senders] - positions[data.receivers]

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(r**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * polynomial_cutoff(
                r_norm,
                self.cutoff,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            r,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(
                features,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )

        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * jax.lax.stop_gradient(
            self.scale
        ) + jax.lax.stop_gradient(self.shift)

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(
            self.atom_energies[data.nodes["species"], None]
        )

        return node_energies

    def __call__(self, data: jraph.GraphsTuple):
        # compute forces as gradient of total energy
        def total_energy_fn(positions: jax.Array, data: jraph.GraphsTuple):
            node_energies = self.node_energies(positions, data).array
            return jnp.sum(node_energies), node_energies

        minus_forces, node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
            data.nodes["positions"], data
        )

        # padded nodes may have nan forces, so we mask them
        node_mask = jraph.get_node_padding_mask(data)[:, None]
        minus_forces = jnp.where(node_mask, minus_forces, 0.0)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        return graph_energies[:, 0], -minus_forces


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    # based on https://github.com/google-deepmind/jraph/blob/51f5990/jraph/_src/models.py#L209-L216
    n_graph = data.n_node.shape[0]
    # equivalent to jnp.sum(n_node), but jittable
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
    )
    return node_gr_idx


def save_model(path: str, model: eqx.Module, config: dict):
    """Save a model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str) -> tuple[Nequip, dict]:
    """Load a model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = Nequip(
            key=jax.random.key(0),
            n_species=len(config["atomic_numbers"]),
            hidden_size=config["hidden_size"],
            lmax=config["lmax"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
        )
        model = eqx.tree_deserialise_leaves(f, model)
        return model, config
