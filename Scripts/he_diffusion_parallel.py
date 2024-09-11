import sys
import os
import json
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes import LammpsParentClass
from mpi4py import MPI

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()


temp = 50

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

save_folder = os.path.join('He_Diffusion_Test')
if not os.path.exists(save_folder):
    os.makedirs(save_folder,exist_ok=True)

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

init_dict['output_folder'] = save_folder

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps()

lmp_class.init_from_box(lmp)

_xyz = init_dict['size'] // 2 + np.array([0.25, 0.5, 0])

lmp.command('create_atoms 3 single %f %f %f' % (_xyz[0], _xyz[1], _xyz[2]))

rng = np.random.randint(low = 1, high = 100000)

lmp.command('group mobile type 2 3')

lmp.command('dump mydump all custom 1000 %s/out.*.atom id type x y z'  % save_folder)

lmp_class.run_MD(lmp, temp = 2 * temp, timestep=1e-3, N_steps=5e6)