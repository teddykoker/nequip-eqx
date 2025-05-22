import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from nequip_eqx.data import default_collate_fn
from nequip_eqx.model import Nequip


def test_model():
    key = jax.random.key(0)

    # small model for testing
    model = Nequip(
        key,
        n_species=1,
        lmax=1,
        hidden_size=8,
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )

    batch = jraph.GraphsTuple(
        n_node=np.array([3]),
        n_edge=np.array([3]),
        nodes={
            "species": np.zeros((3,), dtype=np.int32),
            "positions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ).astype(np.float32),
            "forces": np.ones((3, 3), dtype=np.float32),
        },
        edges=None,
        senders=jnp.array([0, 1, 2]),
        receivers=jnp.array([1, 2, 0]),
        globals={"energy": np.ones((1,))},
    )
    energy, forces = model(batch)
    assert energy.shape == batch.globals["energy"].shape
    assert forces.shape == batch.nodes["forces"].shape

    energy, forces = eqx.filter_jit(model)(batch)
    assert energy.shape == batch.globals["energy"].shape
    assert forces.shape == batch.nodes["forces"].shape

    batch = default_collate_fn([batch, batch])
    energy, forces = model(batch)
    assert energy.shape == batch.globals["energy"].shape
    assert forces.shape == batch.nodes["forces"].shape
