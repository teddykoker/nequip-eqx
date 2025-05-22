import argparse
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import yaml

import wandb
from nequip_eqx.data import DataLoader, Dataset
from nequip_eqx.model import Nequip, load_model, node_graph_idx, save_model


def train_stats(
    train_loader: DataLoader, atom_energies: Optional[list[float]] = None
) -> tuple[float, float, float]:
    """Compute mean and RMS of energy and forces respectively, and average number
    of neighbors. If atom_energies are provided shift the total energy by the
    sum of the isolated atom energies."""
    energies, forces, n_neighbors = [], [], []
    for batch in train_loader:
        batch = jraph.unpad_with_graphs(batch)
        if atom_energies is not None:
            node_e0 = jnp.array(atom_energies)[batch.nodes["species"]]
            graph_e0 = jraph.segment_sum(
                node_e0,
                node_graph_idx(batch),
                num_segments=batch.n_node.shape[0],
                indices_are_sorted=True,
            )
        else:
            graph_e0 = 0
        energies.append((batch.globals["energy"] - graph_e0) / batch.n_node)
        forces.append(batch.nodes["forces"])
        n_neighbors.append(batch.n_edge / batch.n_node)

    mean = np.mean(np.concatenate(energies, axis=0))
    rms = np.sqrt(np.mean(np.concatenate(forces, axis=0) ** 2))
    n_neighbors = np.mean(np.concatenate(n_neighbors, axis=0))
    return mean, rms, n_neighbors


@eqx.filter_jit
def loss(model, batch, energy_weight, force_weight):
    """Return loss and MSE of energy and force in eV and eV/Å respectively"""
    energy, forces = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    # MSE energy
    energy_mse = jnp.sum(
        (energy - batch.globals["energy"]) ** 2 * graph_mask
    ) / jnp.sum(graph_mask)

    # MSE energy per atom (see eq. 30 https://www.nature.com/articles/s41467-023-36329-y)
    energy_mse_per_atom = jnp.sum(
        ((energy - batch.globals["energy"]) / batch.n_node) ** 2 * graph_mask
    ) / jnp.sum(graph_mask)

    # MSE forces
    force_mse = jnp.sum((forces - batch.nodes["forces"]) ** 2 * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    total_loss = energy_weight * energy_mse_per_atom + force_weight * force_mse

    return total_loss, (energy_mse, force_mse)


def evaluate(model, dataloader, energy_weight=1.0, force_weight=1.0):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_loss, total_energy_se, total_force_se, total_count = 0, 0, 0, 0
    for batch in dataloader:
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, (el, fl) = loss(model, batch, energy_weight, force_weight)
        total_loss += val_loss * n_graphs
        total_energy_se += el * n_graphs
        total_force_se += fl * n_graphs
        total_count += n_graphs
    return (
        total_loss / total_count,
        jnp.sqrt(total_energy_se / total_count),
        jnp.sqrt(total_force_se / total_count),
    )


def train(config_path: str):
    """Train a NequIP model from a config file. See configs/3bpa.yaml for an example."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]

    wandb.init(project="nequip-eqx", config=config)

    train_dataset = Dataset(
        file_path=config["train_path"],
        atomic_numbers=config["atomic_numbers"],
        split="train",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )
    val_dataset = Dataset(
        file_path=config["train_path"],
        atomic_numbers=config["atomic_numbers"],
        split="val",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    shift, scale, n_neighbors = train_stats(train_loader, atom_energies)
    print(f"shift: {shift}, scale: {scale}, n_neighbors: {n_neighbors}")

    key = jax.random.key(0)
    model = Nequip(
        key,
        n_species=len(config["atomic_numbers"]),
        hidden_size=config["hidden_size"],
        lmax=config["lmax"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=shift,
        scale=scale,
        avg_n_neighbors=n_neighbors,
        atom_energies=atom_energies,
    )

    ema_model = model

    optim = optax.amsgrad(config["learning_rate"])
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    schedule_state = schedule.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, ema_model, step, opt_state, schedule_state, batch):
        # training step
        (total_loss, (energy_mse, force_mse)), grads = eqx.filter_value_and_grad(
            loss, has_aux=True
        )(model, batch, config["energy_weight"], config["force_weight"])
        updates, opt_state = optim.update(grads, opt_state)
        updates = optax.tree_utils.tree_scalar_mul(schedule_state.scale, updates)
        model = eqx.apply_updates(model, updates)

        # update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)

        return (
            model,
            ema_model,
            opt_state,
            total_loss,
            jnp.sqrt(energy_mse),
            jnp.sqrt(force_mse),
        )

    step = jnp.array(0)
    best_val_loss = float("inf")

    for epoch in range(config["n_epochs"]):
        for batch in train_loader:
            model, ema_model, opt_state, total_loss, energy_mse, force_mse = train_step(
                model, ema_model, step, opt_state, schedule_state, batch
            )
            step = step + 1
            if step % config["log_every"] == 0:
                wandb.log(
                    {
                        "train/loss": total_loss,
                        "train/energy_rmse": energy_mse,
                        "train/force_rmse": force_mse,
                    },
                    step=step,
                )

        val_loss, val_energy_rmse, val_force_rmse = evaluate(
            ema_model, val_loader, config["energy_weight"], config["force_weight"]
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(Path(wandb.run.dir) / "checkpoint.eqx", ema_model, config)

        # update learning rate on validation loss
        _, schedule_state = schedule.update(
            updates=eqx.filter(model, eqx.is_array),
            state=schedule_state,
            value=val_loss,
        )

        wandb.log(
            {
                "val/loss": val_loss,
                "val/energy_rmse": val_energy_rmse,
                "val/force_rmse": val_force_rmse,
                "lr_scale": schedule_state.scale,
                "learning_rate": config["learning_rate"] * schedule_state.scale,
                "epoch": epoch,
            },
            step=step,
        )

        if config["learning_rate"] * schedule_state.scale < 1e-6:
            print("lr < 1e-6, stopping training")
            break

        if schedule_state.count > 1000:
            print("1000 epochs without improvement, stopping training")
            break

    ema_model, _ = load_model(Path(wandb.run.dir) / "checkpoint.eqx")

    for test_file in config["test_paths"]:
        test_dataset = Dataset(
            file_path=test_file,
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
        )
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
        _, test_energy_rmse, test_force_rmse = evaluate(ema_model, test_loader)
        print(f"file: {test_file}")
        print(f"energy RMSE: {test_energy_rmse * 1000:.2f} meV")
        print(f"force RMSE: {test_force_rmse * 1000:.2f} meV/Å")
        wandb.log(
            {
                f"test/{Path(test_file).name}/energy_rmse": test_energy_rmse,
                f"test/{Path(test_file).name}/force_rmse": test_force_rmse,
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
