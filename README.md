# Solvent-Inclusive ML/MM Simulations: Assessments of Structural, Dynamical, and Thermodynamic Accuracy

This repository contains the simulation scripts used to reproduce the results from the manuscript:

>


## Conda Environment

To run the code in this directory, create a conda environment with the required packages using Anaconda or Mamba. Installation must be performed on a GPU node. A few plugins need to be generated from source.

To create the environment using mamba, run the following command:

```bash
module load mamba

mamba env create --name <env_name> --file=environment.yml
```

The command above will install the following packages installed from the `conda-forge`, `nvidia`, and `pytorch` channels:

- `openmm`
- `openmm-torch`
- `openmmtools`
- `openmm-plumed`
- `pytorch`
- `pytorch-cuda=11.6`
- `cudatoolkit=11.4`
- `nnpops`
- `plumed`

### OpenMM Plugins

**OpenMM-DeepMD plugin**:

This plugin must be built from source. Follow the instructions from the [OpenMM-DeepMD plugin repository](https://github.com/JingHuangLab/openmm_deepmd_plugin).


## License
Code is made available under an MIT license. Both of these allow broad reuse with attribution.

## Issues
For any questions, please raise an issue.