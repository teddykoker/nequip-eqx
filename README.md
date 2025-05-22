![molecular dynamics animation](images/md.gif)

*3BPA molecular dynamics using nequip-eqx*.

# nequip-eqx

`nequip-eqx` is a JAX implementation of the neural network interatomic potential
NequIP, introduced by Batzner et al. in [E(3)-equivariant graph neural networks
for data-efficient and accurate interatomic
potentials](https://www.nature.com/articles/s41467-022-29939-5).

The goal of this repository is to offer a simple (<1000 lines of code)
implementation while providing competitive performance to existing
implementations.



## Usage

### Installation

```bash
pip install nequip-eqx
```

### Training

Models are trained with the `nequip_eqx_train` command using a single `.yml`
configuration file:

```bash
nequip_eqx_train <config>.yml
```

See [`configs/3bpa.yml`](configs/3bpa.yml) for an example configuration file for training on the 3BPA dataset. 
Pretrained weights for 3BPA are available in the [`models/`](models/) directory.

```bash
tar -C data -xf data/dataset_3BPA.tar.gz  # decompress data
nequip_eqx_train configs/3bpa.yml # takes ~20 hrs on an NVIDIA RTX A5500
```

### Testing

Models can be evaluated with the `nequip_eqx_test` command by supplying a path
to a pretrained model, and a test `.xyz` file, e.g.:

```bash
nequip_eqx_test \
    --model models/nequip_3bpa.eqx \
    --file data/dataset_3BPA/test_300K.xyz
```

### ASE calculator

Using `nequip_eqx.calculator.NequipCalculator`, you can perform calculations in
ASE with a pre-trained NequIP model.

```python
import ase.io
from ase.md.langevin import Langevin
from ase import units

from nequip_eqx.calculator import NequipCalculator

atoms = ase.io.read("data/dataset_3BPA/test_300K.xyz", index=0, format="extxyz")
atoms.calc = NequipCalculator("models/nequip_3bpa.eqx")

dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,
    temperature_K=300,
    friction=0.01,
    trajectory="md.traj",
)
dyn.run(steps=1000)

```

## Comparison with other codes

In order to verify correctness of the implementation, we compare performance on
the 3BPA dataset to two different PyTorch NequIP implementations: 

1. The results from Musaelian et al. in ["Learning local equivariant
representations for large-scale atomistic
dynamics"](https://www.nature.com/articles/s41467-023-36329-y), using a version
of their [`nequip`](https://github.com/mir-group/nequip) repository.  
2. The
results from Batatia et al. in ["MACE: Higher Order Equivariant Message
Passing Neural Networks for Fast and Accurate Force
Fields"](https://arxiv.org/abs/2206.07697), using a version of their [`mace`](https://github.com/ACEsuit/mace) repository.

We use the same hyperparameters as [1], which can be viewed in [`configs/3bpa.yml`](configs/3bpa.yml), with the following exceptions:

 * Instead of hidden irreps of `64x0e + 64x0o + 64x1o + 64x1e + 64x2e + 64x2o +
    64x3o + 64x3e`, we opt for `128x0e + 128x1o + 128x2e + 128x3o` for simplicity
    with the same feature dimensions.
 * It is not clear what initialization was originally used for the radial MLP,
    but we use Kaiming normal, i.e. sampling from $\mathcal{N}(0, \mathrm{std})$ with $\mathrm{std} =
    \sqrt{\frac{4.0}{\mathrm{fan\\_in}}}$. 
 * Isolated atom energies are added to each predicted node energy.
    
Resulting energy (E) and force (F) RMSE in meV and meV/Å respectively.

| Code     | [`nequip`](https://github.com/mir-group/nequip) | [`mace`](https://github.com/ACEsuit/mace)  | `nequip-eqx` (this repo) |
|--------------|--------------|-------------------|---|
| 300 K E      | 3.3 (0.1)    | 3.1 (0.1)         | **2.9**
| 300 K F | 10.8 (0.2)   | 11.3 (0.2)        | **9.5**
| 600 K E | 11.2 (0.1)   | 11.3 (0.3)       | **10.8**
| 600 K F | 26.4 (0.1)   | 27.3 (0.3)        | **24.2**
| 1200 K E | 38.5 (1.6)   | 40.8 (1.3)        | **34.1**
| 1200 K F | 76.2 (1.1)   | 86.4 (1.5)        | **75.4**

## See also

 * [mariogeiger/nequip-jax](https://github.com/mariogeiger/nequip-jax): Another basic implementation of a NequIP style model in JAX.
 * [e3nn/e3nn-jax](https://github.com/e3nn/e3nn-jax): JAX library for E(3)-equivariant neural networks used in this repo.
 * [patrick-kidger/equinox](https://github.com/patrick-kidger/equinox): JAX library for building neural network architectures, used in this repo.
 

## Citations

```bibtex
@article{batzner20223,
  title={E (3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

```bibtex
@article{musaelian2023learning,
  title={Learning local equivariant representations for large-scale atomistic dynamics},
  author={Musaelian, Albert and Batzner, Simon and Johansson, Anders and Sun, Lixin and Owen, Cameron J and Kornbluth, Mordechai and Kozinsky, Boris},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={579},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

```bibtex
@article{batatia2022mace,
  title={MACE: Higher order equivariant message passing neural networks for fast and accurate force fields},
  author={Batatia, Ilyes and Kovacs, David P and Simm, Gregor and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={11423--11436},
  year={2022}
}
```
