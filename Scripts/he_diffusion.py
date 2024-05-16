import sys
import os
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps
from itertools import combinations

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def maximally_distant_points(points):

    max_distance = 0
    max_pair = None
    
    pbc = 22.009548038010326 - -1.0380103268104445e-06 

    # Generate all pairs of points
    pairs = combinations(points, 2)
    
    # Iterate over pairs and find the pair with maximum distance
    for pair in pairs:

        delta = pair[0] - pair[1]

        delta = delta - np.round(delta/pbc)*pbc
                
        distance = np.linalg.norm(delta)

        if distance > max_distance:
            max_distance = distance
            max_pair = pair
            
    centroid = points[0]

    return centroid, max_pair, max_distance

def check_pbc(centroid, pbc):

    for i in range(len(centroid) - 1):

        disp = centroid[i + 1] - centroid[i]

        for j, _d in enumerate(disp):

            if _d > 0.9*pbc[j]:
                centroid[i + 1, j] -= pbc[j]

            elif _d < -0.9*pbc[j]:
                centroid[i + 1, j] += pbc[j]
    return centroid

def diffusion_coefficient(lmp_class:LammpsParentClass, input_filepath, species_of_interest, temp, timestep = 1e-4, N_steps = 1e6, sample_steps=1e3):

    if not isinstance(species_of_interest, list):
        species_of_interest = [species_of_interest]

    lmp = lammps(name = lmp_class.machine, comm=lmp_class.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.commands_list(lmp_class.init_from_datafile(input_filepath))

    species = np.array(lmp.gather_atoms('type', 0, 1))
    
    lmp_class.N_species = np.array([sum(species == k) for k in range(1, 4)])

    condition = np.logical_or.reduce([species == val for val in species_of_interest])

    atom_idx = np.where(condition)[0]

    N_iterations = int(N_steps//sample_steps)

    data = np.zeros(( N_iterations , len(atom_idx), 3))

    for i in range(N_iterations):

        lmp_class.run_MD(lmp, temp, timestep, sample_steps)

        lmp.command('write_dump all custom %s id type x y z' % os.path.join(lmp_class.output_folder, 'Atom_Files', 'test.%d.atom' % i))
        
        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        xyz = xyz.reshape(len(xyz)//3, 3)

        data[i] = xyz[atom_idx]

    return data

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'He_Diffusion'


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))


init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder


init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp_class.perfect_crystal()

input_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, 0, 0, 0)
    
output_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, 0, 0, 1)

defect_centre = ( lmp_class.alattice*lmp_class.size/2 )*np.ones((3,))

lmp_class.add_defect(input_filepath, output_filepath, 3, 1, defect_centre)

input_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, 0, 0, 1)

displacement = []

temp_arr = np.linspace(1000, 2000, 10)

n_replica = 20

N_steps = 1e5

timestep = 1e-3

diffusion_coef = []

for temp in temp_arr:

    displacement = []

    for _replica in range(n_replica):

        data = diffusion_coefficient(lmp_class, input_filepath, 3, temp, timestep = timestep, N_steps = N_steps, sample_steps=1e2)

        centroid = np.zeros((data.shape[0], 3))

        max_pair = np.zeros((data.shape[0]))

        max_distance = np.zeros((data.shape[0]))

        for i, pt in enumerate(data):

            _centroid, _max_pair, _max_distance = maximally_distant_points(pt)
            
            centroid[i] = _centroid

            max_pair[i] = _max_pair

            max_distance[i] = _max_distance

        centroid = check_pbc(centroid, lmp_class.pbc)

        displacement.append(centroid - centroid[0])

    displacement = np.array(displacement)
    
    displacement_corrected = np.zeros(displacement.shape)

    var = displacement[:,0,:]

    for i in range(1,displacement.shape[1]):    
        var = ( (i - 1)*var + displacement[:,i,:])/i
        displacement_corrected[:, i, :] = displacement[:, i, :] - 2*var

    msd_corrected = np.sum( displacement_corrected**2, axis = 2)

    msd = np.mean(np.sum( displacement**2, axis = 2), axis = 0)

    D_msd = msd/(6*np.arange(1,displacement.shape[1]+1)*timestep*N_steps)

    D_msd_corrected = np.mean(msd_corrected, axis = 0)/(6*np.arange(1,displacement.shape[1]+1)*timestep*N_steps)

    diffusion_coef.append(D_msd_corrected[-1])

    print(temp, diffusion_coef[-1])

print(diffusion_coef)