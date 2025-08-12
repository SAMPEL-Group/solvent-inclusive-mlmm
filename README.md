# Solvent-Inclusive ML/MM Simulations

This repository contains the simulation scripts used to reproduce the results from the manuscript:

> V. Gopal, C. Kirkvold, A. Gordon, J. Goodpaster, and S. Sarupria.  
*Solvent-Inclusive ML/MM Simulations: Assessments of Structural, Dynamical, and Thermodynamic Accuracy.*  
ChemRxiv, preprint (2025).  
[Link pending ChemRxiv approval]

## Software Requirements

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

This plugin must be built from source. Follow the instructions from the [OpenMM-DeepMD plugin repository](https://github.com/JingHuangLab/openmm_deepmd_plugin). This study built the plugin from commit `04603ac`.

**OpenMM-Plumed plugin**:

This plugin will be installed from the `environment.yml` file described above. However, the plugin will not contain the `manyrestraints` module needed to implement the `UWALLS` and `LWALLS` restraints to define the ML-MM boundary. As a result, PLUMED will need to be built from source and then linked to the OpenMM-Plumed plugin. 

Install PLUMED 2.9.0 using these [instructions](https://www.plumed.org/doc-v2.9/user-doc/html/_installation.html) from the PLUMED website. Make sure to include the `manyrestraints` [module](https://www.plumed.org/doc-v2.9/user-doc/html/mymodules.html) when building PLUMED. To ensure that the OpenMM-Plumed plugin is correctly linked to the PLUMED library, the `PLUMED_KERNEL_PATH` environment variable should be modified in the OpenMM python scripts. Find the path to the `lib/libplumedKernel.so` for the PLUMED built from source. Now, in the OpenMM script, change the `PLUMED_KERNEL_PATH`: 

```python
import os

os.environ['PLUMED_KERNEL'] = '<path_to_plumed>/plumed/plumed-2.9.0/lib/libplumedKernel.so'

from openmmplumed import PlumedForce
```
An example of how to do this for an OpenMM script can be seen with the scripts provided.

## DeepMD Models

The DeepMD model used in this work was trained on the data from the following paper: 

>M. de la Puente, R. David, A. Gomez, and D. Laage.  
*Acids at the Edge: Why Nitric and Formic Acid Dissociations at Air–Water Interfaces Depend on Depth and on Interface Specific Area.*  
**Journal of the American Chemical Society** **144** (23), 10524–10529 (2022).  
[https://doi.org/10.1021/jacs.2c03099](https://doi.org/10.1021/jacs.2c03099)

The model was trained using the DeepMD-kit 2.2.9 [package](https://deepmd.readthedocs.io/en/latest/). A trained model is not provided with this repo as the associated training data is not publically available. However, we provide links to publicly available DeepMD models that can be used. Two examples of water models are provided below:

1. [Water model](https://github.com/deepmodeling/dpgen/discussions/699) associated with the paper:  
   >L. Zhang, H. Wang, R. Car, and W. E.  
*Phase Diagram of a Deep Potential Water Model.*  
**Physical Review Letters** **126** (23), 236001 (2021).  
[https://doi.org/10.1103/PhysRevLett.126.236001](https://doi.org/10.1103/PhysRevLett.126.236001)
2. [Water model](https://github.com/paesanilab/Data_Repository/tree/main/Quantum-phase-diagram-of-water/Deep_Neural_Network_Potential) associated with the paper:  
   >S. L. Bore and F. Paesani.  
*Realistic phase diagram of water from “first principles” data-driven quantum simulations.*  
**Nature Communications** **14**, 3349 (2023).  
[https://doi.org/10.1038/s41467-023-38855-1](https://doi.org/10.1038/s41467-023-38855-1)

Use of these models in the ML/MM simulations only requires a change to the name of the model in the OpenMM script.

## Simulation Scripts

We provide OpenMM scripts to run both systems studied in this work: neat water and formic acid in water. The files are located in the `scripts` directory. Comments are provided to explain the different sections of the code, including the custom integrator scheme and the ML/MM boundary definition. 

1. **Neat water system**: `scripts/mlmm_water_system_0.5nm_ml_region.py`

2. **Formic acid in water system**: `scripts/mlmm_fa_system_0.5nm_ml_region.py`