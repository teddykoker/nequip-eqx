import equinox as eqx
import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from nequip_eqx.data import (
    atomic_numbers_to_indices,
    pad_graph_to_nearest_power_of_two,
    preprocess_graph,
)
from nequip_eqx.model import load_model


class NequipCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model, self.config = load_model(model_path)
        self.atom_indices = atomic_numbers_to_indices(self.config["atomic_numbers"])

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        graph = preprocess_graph(atoms, self.atom_indices, self.model.cutoff, False)
        padded_graph = pad_graph_to_nearest_power_of_two(graph)
        energy, forces = eqx.filter_jit(self.model)(padded_graph)
        # take energy and forces without padding
        self.results["energy"] = np.array(energy[0])
        self.results["forces"] = np.array(forces[: len(atoms)])
