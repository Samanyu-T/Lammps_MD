import sys
import os
import numpy as np
import json
from mpi4py import MPI

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import Monte_Carlo_Methods

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, proc_id)

potential_arr = np.linspace(0, 0.5, 2)

temp_arr = np.linspace(600, 910, 7)

replica_id_arr = np.arange(8)

potential = potential_arr[ proc_id % 2]

temp = temp_arr[ proc_id % 7]

replica_id = replica_id_arr[ proc_id % 8]

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)
 
output_folder = os.path.join('Monte_Carlo_HSurface_Test')

init_dict['size'] = 7

# init_dict['orientx'] = [1, 1, 0]

# init_dict['orienty'] = [0, 0, -1]

# init_dict['orientz'] = [-1, 1, 0]

init_dict['orientx'] = [1, 1, 1]
init_dict['orienty'] = [-1,2,-1]
init_dict['orientz'] = [-1,0, 1]

init_dict['surface'] = 20

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = Monte_Carlo_Methods(init_dict, comm_split, proc_id)

lmp.perfect_crystal()

for i in range(5):
    input_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, 0, i, 0)
    
    output_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, 0, i+1, 0)

    ef, rvol = lmp.add_defect(input_filepath, output_filepath, 2, 1, defect_centre=np.array([0, 0, 0])) 

    print(ef, rvol)