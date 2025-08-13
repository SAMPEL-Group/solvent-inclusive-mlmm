import openmmtools
import os
from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from OpenMMDeepmdPlugin import DeepPotentialModel
import itertools
import subprocess
import datetime
import math

# changed the plumed usage from the conda environment to the one in the system
os.environ['PLUMED_KERNEL'] = '/home/sarupria/gopal145/software/plumed/plumed-2.9.0/lib/libplumedKernel.so'
from openmmplumed import PlumedForce

# run shell command of which plumed
# to get the path to the plumed executable
print('Confirm that the plumed used is the one built from source...')
subprocess.run(['which', 'plumed'])
print('\n')

def identify_atoms_in_ml_region(positions, 
                                ml_region_cutoff, 
                                formic_acid_indices, 
                                surrounding_water_indices, 
                                box_vectors,
                                central_water_oxygen_index):
    '''
    This function identifies which atom indices are within the cutoff of the formic acid molecule

    Parameters:
    -----------
    positions : numpy.ndarray
      The positions of all atoms in the system
    ml_region_cutoff : float
      The cutoff distance for identifying atoms within the formic acid molecule
    formic_acid_indices : list
      The indices of the atoms in the formic acid molecule
    surrounding_water_indices : list
      The indices of the atoms in the surrounding water molecules
    box_vectors : numpy.ndarray
      The box vectors defining the periodic boundary conditions
    formic_acid_carbon_indices : list
        The indices of the carbon atom in the formic acid molecule

    Returns:
    --------
    atoms_within_cutoff : numpy.ndarray
      An array of atom indices that are within the cutoff of the formic acid molecule
      while using periodic boundary conditions
    
    '''

    # identify all the atoms that are within the cutoff of the formic acid molecule carbon
    # while using periodic boundary conditions

    # get the positions of the formic acid atoms
    central_water_oxygen_positions = positions[central_water_oxygen_index]
    water_positions = positions[surrounding_water_indices]

    box_length = box_vectors[0][0]
    # confirm that the box is cubic
    assert box_length == box_vectors[1][1] == box_vectors[2][2]
    # ensure other two box vectors components are 0
    assert box_vectors[0][1] == 0 
    assert box_vectors[0][2] == 0

    # use mdanalysis to calculate all the pairwise distances in the system 
    # and identify the atoms within the cutoff

    # convert positions and distances to angstroms b/c MDAnalysis uses angstroms
    central_water_oxygen_positions = central_water_oxygen_positions * 10
    water_positions = water_positions * 10
    ml_region_cutoff = ml_region_cutoff * 10
    box_vectors = box_vectors * 10

    # box dimensions have to be in the format of mdanalysis dimenions are returned by
    # Timestep.dimensions (A B C, alpha, beta, gamma)
    box_dimensions = np.array([box_vectors[0][0], box_vectors[1][1], box_vectors[2][2], 90, 90, 90])

    pairs, pair_distances = capped_distance(central_water_oxygen_positions, 
                                            water_positions, 
                                            max_cutoff=ml_region_cutoff, 
                                            box=box_dimensions)
    # first, we note that the indices in the pairs reflect the indices in the positions arrays supplied to the function
    # and not the original indices in the system. We need to convert these indices to the original indices in the system
    # to do this, we will add the number of atoms in the main water molecule to the second index in the pairs array
    # this will give us the original index of the water atom in the system
    pairs[:, 1] = pairs[:, 1] + len(central_water_oxygen_index)

    # now, we want to identify the indices of the indices of water molecules as a whole that are within the cutoff
    # take the indices from the 2nd column of pairs and remove duplicates
    atoms_within_cutoff = np.array(list(set(pairs[:, 1])))
    # arange in ascending order
    atoms_within_cutoff = np.sort(atoms_within_cutoff)

    # make an array of the water molecule indices, group the surronding water indices into groups of 3
    water_molecule_indices = np.array_split(np.array(range(3,surrounding_water_indices.max()+1)), len(surrounding_water_indices)/3)

    # now, we want to identify the indices of the water molecules with atoms that are within the cutoff
    water_molecules_within_cutoff = []
    for atom_index in atoms_within_cutoff:
        for water_molecule in water_molecule_indices:
            if atom_index in water_molecule:
                water_molecules_within_cutoff.append(water_molecule)

    # retain only the unique water molecules
    water_molecules_within_cutoff = np.unique(water_molecules_within_cutoff, axis=0)

    atoms_within_cutoff = water_molecules_within_cutoff.flatten()

    # the atoms within the cutoff should be in the ml region                
    return atoms_within_cutoff

def calculate_spherical_radius(num_water, num_formic_acid, mass_water=18.015, mass_formic_acid=46.025, density=1.0):
    '''
    Calculate the radius of a spherical volume for a given number of water molecules
    and formic acid molecules, assuming a target density.

    Parameters:
    -----------
    num_water : int
        Number of water molecules.
    num_formic_acid : int
        Number of formic acid molecules.
    mass_water : float
        Molecular mass of water in g/mol. Default is 18.015.
    mass_formic_acid : float
        Molecular mass of formic acid in g/mol. Default is 46.025.
    density : float
        Target density in g/cm^3. Default is 1.0.

    Returns:
    --------
    radius : float
        The radius of the spherical volume in nanometers.
    '''
    # Avogadro's constant
    AVOGADRO_CONSTANT_NA = 6.02214076e23  # molecules/mol
    
    # Calculate the total mass in grams
    total_mass = (num_water * mass_water + num_formic_acid * mass_formic_acid) / AVOGADRO_CONSTANT_NA  # in grams
    
    # Calculate the volume in cm^3
    volume_cm3 = total_mass / density  # in cm³
    
    # Convert volume to Å³ (1 cm³ = 1e24 Å³)
    volume_A3 = volume_cm3 * 1e24  # in Å³
    
    # Convert Å³ to nm³ (1 Å³ = 1e-3 nm³)
    volume_nm3 = volume_A3 * 1e-3  # in nm³
    
    # Calculate the radius of the sphere using V = 4/3 * pi * R³
    radius = ((3 * volume_nm3) / (4 * math.pi))**(1/3)  # in nm
    
    return radius


####################################
# INITIALIZE THE SIMULATION
####################################

# generate the water box system
water = openmmtools.testsystems.WaterBox(model='tip3p', box_edge=3.0*nanometers)
# export the structure to a pdb file
PDBFile.writeFile(water.topology, water.positions, open('water_box_mlmm.pdb', 'w'))

# use the pdf file from the NPT equilibration
water_box = PDBFile('output_waters_run_02_final_frame.pdb')

modeller = Modeller(water_box.topology, water_box.positions)

# create a numpy array with the formic acid indices and the water indices
total_num_atoms = modeller.topology.getNumAtoms()
waterAtomIndices = np.arange(0, total_num_atoms)

num_central_oxygens = 1
central_oxygen_index = np.array([0])
remaining_atom_indices = np.array(list(range(1, total_num_atoms)))
assert len(central_oxygen_index) + len(remaining_atom_indices) == total_num_atoms

# get the current positions of the atoms as a numpy array
# to get the values as a numpy array without simtk units, we need to divide by the unit
# we also enforce that the center of every molecule lies in the same periodic box
current_positions = modeller.positions.value_in_unit(nanometer)
current_positions = np.array(current_positions)

box_vectors = modeller.topology.getPeriodicBoxVectors().value_in_unit(nanometer)
box_vectors = np.array(box_vectors)

# identify the atoms within the cutoff of the formic acid molecule
ml_region_cutoff = 0.5 # nm
atoms_within_cutoff = identify_atoms_in_ml_region(
    current_positions, 
    ml_region_cutoff, 
    central_oxygen_index, 
    remaining_atom_indices, 
    box_vectors,
    central_oxygen_index)


# the ml region atoms will be both the atoms within the cutoff and the
# central water molecule, excludes central oxygen
atoms_within_cutoff = np.array(atoms_within_cutoff.tolist() + [1, 2])
atoms_within_cutoff = np.sort(atoms_within_cutoff)
print('Atoms within the cutoff:', atoms_within_cutoff)

ml_region_particles = atoms_within_cutoff.tolist() + central_oxygen_index.tolist()
# sort ml_region_particles
ml_region_particles = np.sort(ml_region_particles)
# assert that the ml_region_particles is divisible by 3
assert len(ml_region_particles) % 3 == 0
non_ml_region_particles = np.setdiff1d(np.arange(total_num_atoms), ml_region_particles)

# assert there is no overlap between the ml region and non-ml region particles
assert len(np.intersect1d(ml_region_particles, non_ml_region_particles)) == 0

# now, we will create an OpenMM system object from scratch 
# we will first add the DeepMD force to the ML system atoms 

system = System()

##### DEEPMD FORCE ########

# Set up the dp_system with the dp_model.    
dp_model_file = os.path.join('LR_model.pb')
dp_model = DeepPotentialModel(dp_model_file)
coord_coeff = 10 # convert from to nm to Angstroms
force_coeff = 964.8792534459 # convert from kJ/(mol*nm) to kcal/(mol*A)
energy_coeff = 96.48792534459 # convert from eV to kJ/mol
dp_model.setUnitTransformCoefficients(coord_coeff, force_coeff, energy_coeff)
dp_force = dp_model.dp_force

for atom in modeller.topology.atoms():
    if atom.index in ml_region_particles:
        atom_type = atom.element.symbol
        # add the particle to the system
        system.addParticle(atom.element.mass)    
        dp_force.addParticle(atom.index, atom_type)

# make dp_force part of force group 1
dp_force.setForceGroup(1)
# add the dp_force to the system
system.addForce(dp_force)

##### vdW AND ELECTROSTATIC FORCES - NOBONDED FORCE ########

# now, we add the non-ml region particles to the system with the TIP3P force field
# for water. We will also add constraints to the water molecules

# create a NonbondedForce object
nonbonded_force = NonbondedForce()
nonbonded_force.setNonbondedMethod(NonbondedForce.PME)
nonbonded_force.setCutoffDistance(1.0 * nanometer)

# add the water atoms to the system
for atom in modeller.topology.atoms():
   
    if atom.index in ml_region_particles:
        atom_type = atom.element.symbol

        # add the Lennard-Jones parameters for the particle
        if atom_type == 'O':
            sigma = 0.3151 * nanometer
            epsilon = 0.63627 * kilojoules_per_mole
            charge = -0.834
        elif atom_type == 'H':
            sigma = 0.0 * nanometer
            epsilon = 0.0 * kilojoules_per_mole
            charge = 0.417
        else:
            raise ValueError('Atom type not recognized!')
        nonbonded_force.addParticle(charge, sigma, epsilon)

    if atom.index in non_ml_region_particles:
        atom_type = atom.element.symbol

        system.addParticle(atom.element.mass)

        # add the Lennard-Jones parameters for the particle
        if atom_type == 'O':
            sigma = 0.3151 * nanometer
            epsilon = 0.63627 * kilojoules_per_mole
            charge = -0.834
        elif atom_type == 'H':
            sigma = 0.0 * nanometer
            epsilon = 0.0 * kilojoules_per_mole
            charge = 0.417
        else:
            raise ValueError('Atom type not recognized!')
        nonbonded_force.addParticle(charge, sigma, epsilon)

# create exceptions between all pairs of ML region particles
for i in ml_region_particles:
    for j in ml_region_particles[i:]:
        nonbonded_force.addException(i, j, 0, 1, 0)

# make nonbonded force part of force group 2
nonbonded_force.setForceGroup(2)
system.addForce(nonbonded_force)

# ### CONSTRAINTS ########

# loop through the non-ml region atoms in sets of 3
# then find the combination of all the pairs of atoms in the set of 3
for i in range(0, len(non_ml_region_particles), 3):
    for pair in itertools.combinations(non_ml_region_particles[i:i+3], 2):
       
        # add constraint between the pairs with value based on atom type
        if pair[0] in waterAtomIndices and pair[1] in waterAtomIndices:

            # check what atom types each atom is 
            atom_1 = list(modeller.topology.atoms())[pair[0]]
            atom_2 = list(modeller.topology.atoms())[pair[1]]

            if atom_1.element._name == 'hydrogen' and atom_2.element._name == 'oxygen':
                # add a constraint between the hydrogen and oxygen atoms
                system.addConstraint(pair[0], pair[1], 0.09572 * nanometer)
            
            elif atom_1.element._name == 'oxygen' and atom_2.element._name == 'hydrogen':
                # add a constraint between the hydrogen and oxygen atoms
                system.addConstraint(pair[0], pair[1], 0.09572 * nanometer)
            
            elif atom_1.element._name == 'hydrogen' and atom_2.element._name == 'hydrogen':

                # add a constraint between the hydrogen and hydrogen atoms
                system.addConstraint(pair[0], pair[1], 0.15139 * nanometer)

####################################

# DeepMD force is part of force group 1
# Nonbonded force is part of force group 2
# constraints are part of force group 0
# Plumed force is part of force group 0

# this separation of forces into different force groups will allow us to
# zero-out specific forces in the next section

####################################
# DEFINE CUSTOM INTEGRATOR
####################################

# Define your parameters and time step.
dt = 0.25 * femtosecond
friction = 5.0 / picosecond
temperature = 300 * kelvin
# kB = BOLTZMANN_CONSTANT_kB  # Boltzmann constant in OpenMM units
kB = MOLAR_GAS_CONSTANT_R

integrator = CustomIntegrator(dt)
integrator.addPerDofVariable('fml', 0.0)
integrator.addPerDofVariable('include_ml', 0) # default is 0, no atoms are in the ml region
is_ml = np.zeros([total_num_atoms, 3]) # change the perDofVariable to reflect the atom indices in the ml region
is_ml[ml_region_particles] = 1
integrator.setPerDofVariableByName('include_ml', is_ml)
integrator.addPerDofVariable('fmm', 0.0)
integrator.addPerDofVariable('include_mm', 0) # default is 0, no atoms are in the mm region
is_mm = np.zeros([total_num_atoms, 3]) # change the perDofVariable to reflect the atom indices in the mm region
is_mm[non_ml_region_particles] = 1
integrator.setPerDofVariableByName('include_mm', is_mm)
integrator.addGlobalVariable("a", np.exp(-friction*dt))
integrator.addGlobalVariable("b", np.sqrt(1-np.exp(-2*friction*dt)))
integrator.addGlobalVariable("kT", kB*temperature)
integrator.addPerDofVariable("x1", 0)
integrator.addUpdateContextState()
integrator.addComputePerDof("fml", "f1") # store ml forces in fml which is part of force group 1 ("f1")
integrator.addComputePerDof("fmm", "f2") # store mm forces in fmm which is part of force group 2 ("f2")
integrator.addComputePerDof("v", "v + dt*(((include_ml*fml)+((include_mm)*fmm)+f0)/m)")
# integrator.addComputePerDof("v", "v + dt*f/m")
integrator.addConstrainVelocities()
integrator.addComputePerDof("x", "x + 0.5*dt*v")
integrator.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
integrator.addComputePerDof("x", "x + 0.5*dt*v")
integrator.addComputePerDof("x1", "x")
integrator.addConstrainPositions()
integrator.addComputePerDof("v", "v + (x-x1)/dt")
####################################

platform = Platform.getPlatformByName('CUDA')

simulation = Simulation(
  modeller.topology,
  system,
  integrator,
  platform)

# # load from the checkpoint file
# simulation.loadCheckpoint('prod_gaff_formic_acid_solvated_nvt_spherical_boundary_ml_run3.chk')

# save the initial positions to a pdb file
PDBFile.writeFile(
    simulation.topology,
    simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
    open('prod_mlmm_water_box_bottoms_up_approach_initial_frame.pdb', 'w'))

# print the number of particles in the system
print('Number of particles in the system:', system.getNumParticles())

# set initial positions to the formic acid and water box

# get a subset of positions for the ml region particles from the modeller positions
positions = modeller.positions
simulation.context.setPositions(positions)
# set the box vectors to the simulation
simulation.context.setPeriodicBoxVectors(*box_vectors)

# add COM motion remover
system.addForce(CMMotionRemover())

####################################
# ADD PLUMED FORCE- SPHERICAL RESTRAINT AND COORDINATION NUMBER
####################################

# with the atoms within the cutoff, we can now create a plumed file 
# that will create a spherical restraining potential that will 
# keep atoms inside the ml region to stay there 
# and atoms outside the ml region to stay outside

# identify the list of water atoms that are not the formic acid molecule
# and the atoms within the cutoff
all_atom_indices = np.array(list(range(total_num_atoms)))
waterAtomsOutsideCutoff = np.setdiff1d(all_atom_indices, ml_region_particles)

# sort the atoms within the cutoff
atoms_within_cutoff = np.sort(atoms_within_cutoff)
print ('Atoms within the cutoff:', atoms_within_cutoff)

# store the ML region atoms to a file
atoms_to_save = np.sort(ml_region_particles)
np.savetxt('ml_region_atoms_mlmm_simulation_900_waters_fa_0.5nm_shell_run01.txt', atoms_to_save, fmt='%d')

# assert that the atoms within the cutoff are not in the atoms outside the cutoff
assert len(np.intersect1d(ml_region_particles, waterAtomsOutsideCutoff)) == 0

# identify hydrogen atom indices, the difference between the ml region particles
# and the oxygen atom indices
# loop through the openmm particles and identify the hydrogen atom indices in the entire simulation box

# hydrogen_atom_indices = []
# formic_acid_oxygen_atom_indices = []
# carbon_atom_indices = []
# for atom in simulation.topology.atoms():
#     # if atom is hydrogen and is part of UNL residue, then add to the hydrogen atom indices
#     if atom.element._name == 'hydrogen':
#         if atom.residue.name == 'UNL' and atom.name == 'H1':
#             hydrogen_atom_indices.append(atom.index)
#         elif atom.residue.name != 'UNL':
#             hydrogen_atom_indices.append(atom.index)
#     # if atom is oxygen and is part of UNL residue, then add to the oxygen atom indices
#     if atom.element._name == 'oxygen' and atom.residue.name == 'UNL':
#         formic_acid_oxygen_atom_indices.append(atom.index)
#     if atom.element._name == 'carbon':
#         carbon_atom_indices.append(atom.index)
# hydrogen_atom_indices = np.array(hydrogen_atom_indices)
# formic_acid_oxygen_atom_indices = np.array(formic_acid_oxygen_atom_indices)
# carbon_atom_indices = np.array(carbon_atom_indices)

# the spherical cavity radius should be such that the density of the ML region
# is close to 1.0 g/cm^3
num_waters = (len(ml_region_particles)) // 3
num_formic_acid = 0

spherical_cavity_radius = calculate_spherical_radius(num_waters, num_formic_acid)
spherical_cavity_radius = round(spherical_cavity_radius, 3)

#### plumed force section #########
num_production_steps = 400000

# plumed index starts from 1
script = f'''
UNITS LENGTH=nm TIME=ps

# define the ML central atoms
central_water_oxygen_atom: GROUP ATOMS={','.join(map(str, central_oxygen_index+1))}

# ml region indices
ml_region_atoms: GROUP ...
    ATOMS={','.join(map(str, atoms_within_cutoff+1))}
...

# atoms outside the ml region
water_atoms_outside_cutoff: GROUP ...
    ATOMS={','.join(map(str, waterAtomsOutsideCutoff+1))}
...

# calculate the distances between the COM of the ml central water oxygen and the ml region
d: DISTANCES GROUPA=central_water_oxygen_atom GROUPB=ml_region_atoms 

# create a spherical restraint
restraint_inside: UWALLS ...
    DATA=d 
    AT={spherical_cavity_radius}
    KAPPA=2000.0 
    OFFSET=0.0
    EXP=2.0
    EPS=1
...

# calculate distance between the COM of the ml central water and the atoms outside the ml region
d_outside: DISTANCES GROUPA=central_water_oxygen_atom, GROUPB=water_atoms_outside_cutoff

# create a spherical restraint for the atoms outside the ml region
restraint_outside: LWALLS ...
    DATA=d_outside
    AT={spherical_cavity_radius}
    KAPPA=2000.0
    OFFSET=0.0
    EXP=2.0
    EPS=1
... 

# # print coordination number
# PRINT ARG=c1 STRIDE=4 FILE=colvar_0.5nm_shell_run01.dat
'''
system.addForce(PlumedForce(script))
print('Plumed force added!')
####################################
# print box vectors
print('Box vectors:', simulation.topology.getPeriodicBoxVectors())

# print box vectors
print('Box vectors:', simulation.topology.getPeriodicBoxVectors())

# print the number of forces after adding the plumed force
print('Reinitialized the simulation context!')
print('Number of forces after adding the plumed force:', system.getNumForces())
assert system.getNumForces() == 4 # 1. CMMotionRemover 
                                  # 2. DeepMDForce, 
                                  # 3. PlumedForce,
                                  # 4. NonbondedForce

# save the initial configuration to a pdb file 
PDBFile.writeFile(simulation.topology,
                  simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
                  open('prod_mlmm_waters_0.5nm_shell_run01_initial_frame.pdb', 'w'))

simulation.context.setVelocitiesToTemperature(temperature)
box_vectors = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)/nanometer


# create output files
outfile = os.path.join('prod_mlmm_waters_0.5nm_shell_run01.txt')
dcdfile = os.path.join('prod_mlmm_waters_0.5nm_shell_run01.dcd')
pdbfile = os.path.join('prod_mlmm_waters_0.5nm_shell_run01.pdb')
checkpointfile = os.path.join('prod_mlmm_waters_0.5nm_shell_run01.chk')

simulation.reporters.append(
        DCDReporter(
            dcdfile, 
            40))
simulation.reporters.append(
        StateDataReporter(
            outfile,
            40,
            time=True,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True, 
            temperature=True,
            volume=True,
            density=True, 
            speed=True,))
# create a checkpoint reporter
simulation.reporters.append(
        CheckpointReporter(
            checkpointfile,
            5000))

print('Starting equilibration...')
simulation.step(40000) # 10 ps
print('Finished equilibration!')

simulation.reporters.append(
        PDBReporter(
            pdbfile, 
            2))

print('Starting production...')
simulation.step(400000)

# write out the final configuration to a pdb file
state = simulation.context.getState(getPositions=True,  enforcePeriodicBox=True)
positions = state.getPositions()
# update the topology with the new box vectors
simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simulation.topology, positions,
                open('prod_mlmm_water_box_0.5nm_shell_final.pdb', 'w')
                )

print('Finished production!')
