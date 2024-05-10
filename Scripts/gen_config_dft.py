import sys
import os
import numpy as np
import json
from mpi4py import MPI
from glob import glob

sys.path.append(os.path.join(os.getcwd(), 'git_folder','Classes'))

print(os.path.join(os.getcwd(), 'Classes'))

from Lammps_Classes import Monte_Carlo_Methods

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_Files_DFT_4x4'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 4

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = Monte_Carlo_Methods(init_dict, comm, proc_id)
lmp.perfect_crystal()

struct_lst = []
pe_lst = []
rvol_lst = []
xyz_lst = []

p_events_dict = {'displace':0.75, 'exchange':0.25, 'delete':0, 'create':0}

x = np.arange(3)
y = np.arange(5)
z = np.arange(5)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

dft_xyz =  np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).reshape(-1, 3)

for xyz in dft_xyz:
    
    file = 'Lammps_Files_4x4/Data_Files/V%dH%dHe%d.data' % (xyz[0], xyz[1], xyz[2])

    filename = os.path.basename(file).split('.')[0]

    v = int(filename[1])

    h = int(filename[3])

    he = int(filename[-1])
    
    _ = lmp.hybrid_monte_carlo(file, [2,3], 1000, p_events_dict, 20000, 50,
                    temp = 800, potential = 0, max_displacement = np.ones((3,)),
                    region_of_interest= None, save_xyz=False, diag = False, fix_aniso=True)
