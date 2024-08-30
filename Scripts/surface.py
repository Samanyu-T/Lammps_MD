import sys
import os
import json
import numpy as np
# from mpi4py import MPI
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes_Serial import LammpsParentClass

# comm = MPI.COMM_WORLD

# proc_id = comm.Get_rank()

# n_procs = comm.Get_size()

comm = 0

proc_id = 0

n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_Surface'


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

def surface_binding(lmp, depth):

    pe_arr = np.zeros(depth.shape)

    for i, _d in enumerate(depth):
        pe, _ = lmp.add_defect(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'), 
                               os.path.join(output_folder, 'Data_Files', 'test.data'), 3, 1, np.array([0, 0, _d]))
        
        print(_d, pe - lmp.pe_perfect)
        
        pe_arr[i] = pe

    pe_arr -= lmp.pe_perfect

    return lmp, pe_arr



init_dict['size'] = 6

init_dict['surface'] = 10

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.he'
init_dict['potfile'] = 'git_folder/Potentials/init.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 5, 5)

lmp = LammpsParentClass(init_dict, comm, proc_id)

lmp.perfect_crystal()

lmp, pe_100 = surface_binding(lmp, depth)

lmp.orientx = [1, 1, 0]

lmp.orienty = [0, 0, 1]

lmp.orientz = [1, -1, 0]

lmp.perfect_crystal()

lmp, pe_110 = surface_binding(lmp, depth)


lmp.orientx = [1, 1, 1]

lmp.orienty = [1, -1, 0]

lmp.orientz = [1, 1, -2]

lmp.perfect_crystal()

lmp, pe_111 = surface_binding(lmp, depth)


plt.plot(depth, pe_100, label='100', linestyle=':', marker='o')

plt.plot(depth, pe_110, label='110', linestyle=':', marker='o')

plt.plot(depth, pe_111, label='111', linestyle=':', marker='o')

plt.legend()

plt.xlabel('depth / A')

plt.ylabel('Formation Energy / eV')

plt.title('Formation energy of He interstitial to Tungsten Surfaces')

plt.show()