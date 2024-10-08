import sys
import os
import json
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes import LammpsParentClass
from mpi4py import MPI
import time
import sys
comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, n_procs)

n_temp = 28

temp_arr = np.linspace(10, 300, n_temp)

n_replica = n_procs // n_temp

temp_id = proc_id // n_replica

replica_id = proc_id % n_replica

output_folder = 'He_Diffusion_He2'

if proc_id == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
comm.barrier()


init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 5

init_dict['surface'] = 0

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

temp = temp_arr[temp_id]

save_folder = os.path.join(output_folder,'Temp_%d' % temp)

if replica_id == 0:
    if not os.path.exists(save_folder):
        os.makedirs(save_folder,exist_ok=True)
comm.Barrier()


print(save_folder, temp, temp_id, replica_id, n_replica)
sys.stdout.flush()

n_iterations = 10

for _iterations in range(n_iterations):

    t1 = time.perf_counter()

    lcl_replica_id = replica_id + _iterations * n_replica
    
    init_dict['output_folder'] = output_folder

    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps(comm=comm_split,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    _xyz = init_dict['size'] // 2 + np.array([0.25, 0.35, 0])

    lmp.command('create_atoms 3 single %f %f %f' % (_xyz[0], _xyz[1], _xyz[2]))

    _xyz = init_dict['size'] // 2 + np.array([0.75, 0.5, 0])

    lmp.command('create_atoms 3 single %f %f %f' % (_xyz[0], _xyz[1], _xyz[2]))

    rng = np.random.randint(low = 1, high = 100000)

    lmp.command('group mobile type 2 3')

    lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rng))

    lmp.command('fix fix_temp all nve')

    lmp.command('compute clusters mobile chunk/atom type')

    lmp.command('compute msd_mobile mobile msd/chunk clusters')

    lmp.command('timestep 1e-3')

    n_steps = 10000

    msd = np.zeros((n_steps,))

    for _steps in range(n_steps):
        
        lmp.command('run %d' % (1e3))

        _msd = np.copy(lmp.numpy.extract_compute('msd_mobile', 0, 2))

        msd[_steps] = _msd[-1, -1]

    t2 = time.perf_counter()

    print(t2 - t1, n_steps)
    sys.stdout.flush()

    np.save('%s/msd_data_%d.npy' % (save_folder, lcl_replica_id), msd)
