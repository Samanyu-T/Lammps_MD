import sys
import os
import json
import numpy as np
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes import LammpsParentClass
from mpi4py import MPI
import time, sys, glob

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

n_temp = 10

n_vac = 5

n_he = 5 

n_replica = 10

temp_base = np.linspace(300, 1200, n_temp)

vac_base = np.linspace(0, 1, n_vac)

he_base = np.linspace(0.1, 1, n_he)

xx, yy, zz = np.meshgrid(vac_base, he_base, temp_base)

param_base = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

param_arr = np.vstack([param_base for i in range(n_replica)])

output_folder = 'He_Diffusion_Large'

if proc_id == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
comm.barrier()

np.savetxt('%s/param_arr.txt' % output_folder, param_arr)

init_iteration = len(glob.glob('%s/msd_data_*.npy' % output_folder))

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 38

init_dict['surface'] = 0

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

if init_iteration == len(param_arr):
    exit()

for _iteration in range(init_iteration, len(param_arr)):

    vac, he, temp = param_arr[_iteration]

    t1 = time.perf_counter()
    
    init_dict['output_folder'] = output_folder

    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps(comm=comm)#,cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    natoms = lmp.get_natoms()

    rng = None
    if proc_id == 0:
        rng = np.random.randint(low = 1, high = 100000)
    comm.barrier()
    rng = comm.bcast(rng, root=0)

    lmp.command('delete_atoms random count %d yes all NULL %d' % (int(natoms * vac / 100), rng))

    rng = None
    if proc_id == 0:
        rng = np.random.randint(low = 1, high = 100000)
    comm.barrier()
    rng = comm.bcast(rng, root=0)

    lmp.command('create_atoms %s random %d %d r_simbox overlap 0.5 maxtry 1000' % (3, int(natoms * he / 100), rng))

    lmp.command('group mobile type 2 3')

    lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rng))

    lmp.command('fix 1 all nvt temp 300.0 300.0 100.0')

    lmp.command('compute msd_mobile mobile msd com no average yes')

    lmp.command('timestep 1e-3')

    n_steps = 10000

    msd = np.zeros((n_steps,))

    for _steps in range(n_steps):
        
        lmp.command('run %d' % (1e3))

        _msd = np.copy(lmp.numpy.extract_compute('msd_mobile', 0, 1))

        msd[_steps] = _msd[-1]

    t2 = time.perf_counter()

    print(t2 - t1, n_steps)
    sys.stdout.flush()

    np.save('%s/msd_data_%d.npy' % (output_folder, _iteration), msd)
