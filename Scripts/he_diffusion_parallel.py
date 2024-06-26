import sys
import os
import json
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes_Serial import LammpsParentClass
from mpi4py import MPI

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, n_procs)

n_temp = 14

temp_arr = np.linspace(100, 2000, n_temp)

n_replica = n_procs // n_temp

temp_id = proc_id // n_temp

output_folder = 'He_Diffusion'

if proc_id == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
comm.barrier()

for i in range(10):
    replica_id = proc_id % n_replica + i * n_replica

    temp = temp_arr[temp_id]

    init_dict = {}

    with open('init_param.json', 'r') as file:
        init_dict = json.load(file)

    save_folder = os.path.join(output_folder,'Temp_%d' % temp , 'Iteration_%d' % replica_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)

    init_dict['orientx'] = [1, 0, 0]

    init_dict['orienty'] = [0, 1, 0]

    init_dict['orientz'] = [0, 0, 1]

    init_dict['size'] = 7

    init_dict['surface'] = 0

    init_dict['output_folder'] = output_folder

    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps(comm=comm_split,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.commands_list(lmp_class.init_from_box()) 

    _xyz = init_dict['size'] // 2 + np.array([0.25, 0.5, 0])

    lmp.command('create_atoms 3 single %f %f %f' % (_xyz[0], _xyz[1], _xyz[2]))

    rng = np.random.randint(low = 1, high = 100000)

    lmp.command('group mobile type 2 3')

    lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rng))

    lmp.command('fix fix_temp all nve')

    lmp.command('dump mydump mobile custom 1000 %s/out.*.atom id type x y z'  % save_folder)

    lmp.command('timestep 1e-3')

    lmp.command('run %d' % (5e6))