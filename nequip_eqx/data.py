import pickle
from pathlib import Path

import ase
import ase.io
import ase.neighborlist
import jraph
import numpy as np
from tqdm import tqdm


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    cutoff: float,
    targets: bool,
) -> jraph.GraphsTuple:
    """Preprocess ase.Atoms object into a jraph.GraphsTuple, with optional targets"""
    src, dst = ase.neighborlist.primitive_neighbor_list(
        "ij", atoms.pbc, atoms.cell, atoms.positions, cutoff
    )
    forces = atoms.get_forces().astype(np.float32) if targets else None
    energy = (
        np.array([atoms.get_potential_energy()]).astype(np.float32) if targets else None
    )
    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)]).astype(np.int32),
        n_edge=np.array([len(src)]).astype(np.int32),
        nodes={
            "species": np.array(
                [atom_indices[n] for n in atoms.get_atomic_numbers()],
            ).astype(np.int32),
            "positions": atoms.positions.astype(np.float32),
            "forces": forces,
        },
        edges=None,
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={"energy": energy},
    )


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


# pytorch-like dataset that reads xyz files and returns jraph.GraphsTuple
class Dataset:
    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        split: str = None,
        cutoff: float = 5.0,
        valid_frac: float = 0.1,
        seed: int = 42,
    ):
        self.cutoff = cutoff
        file_path = Path(file_path)
        assert file_path.exists(), f"file path {file_path} does not exist"
        cache_path = file_path.parent / f"{file_path.stem}_cutoff_{cutoff}.pkl"
        if not cache_path.exists():
            data = ase.io.read(file_path, index=":", format="extxyz")
            atomic_indices = atomic_numbers_to_indices(atomic_numbers)
            print("preprocessing graphs...")
            self.graphs = [
                preprocess_graph(atoms, atomic_indices, cutoff, True)
                for atoms in tqdm(data)
            ]
            with open(cache_path, "wb") as f:
                pickle.dump(self.graphs, f)
        else:
            print("loading graphs...")
            with open(cache_path, "rb") as f:
                self.graphs = pickle.load(f)

        if split is not None:
            rng = np.random.RandomState(seed=seed)
            perm = rng.permutation(len(self.graphs))
            train_idx, valid_idx = np.split(perm, [int(len(perm) * (1 - valid_frac))])
            if split == "train":
                self.graphs = [self.graphs[i] for i in train_idx]
            elif split == "val":
                self.graphs = [self.graphs[i] for i in valid_idx]
            else:
                raise ValueError('Split must be "train" or "val"')

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        return self.graphs[idx]


# simple pytorch-like dataloader that batches and pads jraph.GraphsTuple
class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        seed=0,
        shuffle=False,
        drop_last=False,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        self.idx = 0
        self.collate_fn = collate_fn or default_collate_fn

    def __next__(self):
        graphs = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.dataset):
                if len(graphs) > 0 and not self.drop_last:
                    # return partial batch
                    return self.collate_fn(graphs)
                else:
                    raise StopIteration
            graphs.append(self.dataset[self.idxs[self.idx]])
            self.idx += 1
        return self.collate_fn(graphs)

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        return self


def default_collate_fn(graphs):
    # NB: using batch_np is considerably faster than batch with jax
    return pad_graph_to_nearest_power_of_two(jraph.batch_np(graphs))


# from https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/train.py#L117
# NB: using numpy instead of jax.numpy can be orders of magnitude faster for some reason
def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple, _np=np
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
      7 nodes --> 8 nodes (2^3)
      5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
      8 nodes --> 9 nodes
      3 graphs --> 4 graphs

    Args:
      graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
      A graphs_tuple batched to the nearest power of two.
    """

    def _nearest_bigger_power_of_two(x: int) -> int:
        y = 2
        while y < x:
            y *= 2
        return y

    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(_np.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(_np.sum(graphs_tuple.n_edge))
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )
