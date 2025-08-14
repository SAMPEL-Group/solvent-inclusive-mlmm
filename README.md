# Solvent-Inclusive ML/MM Simulations

This repository contains the simulation scripts used to reproduce the results from the manuscript:

> Gopal V, Kirkvold C, Gordon A, Goodpaster J, Sarupria S. Solvent-Inclusive ML/MM Simulations: Assessments of Structural, Dynamical, and Thermodynamic Accuracy. ChemRxiv. 2025; doi:[10.26434/chemrxiv-2025-dlr6c](https://doi.org/10.26434/chemrxiv-2025-dlr6c)

## Software Dependencies

The simulation scripts use OpenMM and the OpenMM-DeePMD plugin, which enables the use of machine-learned interatomic potentials (MLIPs) in molecular dynamics simulations.

### Conda Environment

Most software dependencies can be installed by creating a conda environment using the provided `environment.yml` file. Installation should be performed on a GPU-enabled node.

To create the environment using mamba, run the following command:

```bash
module load mamba

mamba env create --name <env_name> --file=environment.yml
```

The command above will install the following packages:

- `openmm`
- `openmmtools`
- `openmm-plumed`
- `openmmforcefields`
- `openff-toolkit`
- `cudatoolkit`

### OpenMM Plugins

Some OpenMM plugins are required to run the simulations.

**OpenMM-DeePMD plugin**:

This plugin must be built from source. Follow the instructions from the [OpenMM-DeePMD plugin repository](https://github.com/JingHuangLab/openmm_deepmd_plugin). This study built the plugin from commit `04603ac`.

**OpenMM-Plumed plugin**:

This plugin will be installed from the `environment.yml` file described above. However, the plugin will not contain the `manyrestraints` module needed to implement the `UWALLS` and `LWALLS` restraints to define the ML-MM boundary (Eqs. 4 & 5 in the main text). As a result, PLUMED will need to be built from source and then linked to the OpenMM-Plumed plugin. 

Install PLUMED 2.9.0 using these [instructions](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html) from the PLUMED website. Make sure to include the `manyrestraints` [module](https://www.plumed.org/doc-v2.9/user-doc/html/mymodules.html) when building PLUMED. 


## Machine-Learned Interatomic Potentials

The DeepPot-SE models used in this work were trained on data from the following paper: 

>Miguel de la Puente, Rolf David, Axel Gomez, and Damien Laage
*Journal of the American Chemical Society* 2022 144 (23), 10524-10529
DOI: [10.1021/jacs.2c03099](https://doi.org/10.1021/jacs.2c03099)

The models were trained using the DeePMD-kit 2.2.9 [package](https://deepmd.readthedocs.io/en/latest/). 

We provide the two DeePMD model files used in this work in the `scripts` directory. The models are in the `.pb` format, which is compatible with the OpenMM-DeePMD plugin:
- `LR_model.pb`: Referred to as LR model in the main text
- `SR_model.pb`: Referred to as SR model in the main text

## Simulation Scripts

We provide OpenMM scripts to run ML/MM simulations of both systems studied in this work: neat water and formic acid in water. The files are located in the `scripts` directory. Comments are provided to explain the different sections of the code, including the custom integrator scheme and the ML/MM boundary definition. 

1. **Neat water system**: `scripts/mlmm_water_system_0.5nm_ml_region.py`

2. **Formic acid in water system**: `scripts/mlmm_fa_system_0.5nm_ml_region.py`

To run the simulations, load the conda environment created above and execute the script using Python:

```python
python <script_name>.py
```

To ensure that the OpenMM-Plumed plugin is correctly linked to the PLUMED library, the `PLUMED_KERNEL_PATH` environment variable should be modified in the scripts. Find the path to the `lib/libplumedKernel.so` for the PLUMED built from source. In the OpenMM script, change the `PLUMED_KERNEL_PATH`: 

```python
import os

os.environ['PLUMED_KERNEL'] = '<path_to_plumed>/plumed/plumed-2.9.0/lib/libplumedKernel.so'

from openmmplumed import PlumedForce
```
