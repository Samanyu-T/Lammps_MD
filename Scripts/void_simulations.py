import sys
import os
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

# comm = 0

# proc_id = 0

# n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Void_Simulations'

if proc_id == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))

comm.barrier()

init_dict['size'] = 12

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp_class.init_from_box(lmp)

void_radius = 10 / init_dict['alattice']

centre = init_dict['size']/2

lmp.command('region void_region sphere %f %f %f %f side in units lattice' % (centre, centre, centre, void_radius))

lmp.command('group void_group region void_region')

lmp.command('delete_atoms group void_group')

n_h = 3e-2 * lmp.get_natoms()

n_he = 3e-2 * lmp.get_natoms()

rng=None
if proc_id == 0:
    rng = np.random.randint(low = 1, high = 100000)
comm.barrier()

rng = comm.bcast(rng, root=0)
lmp.command('create_atoms 3 random %d %d void_region overlap 0.5 maxtry 50' % (n_he, rng))


rng=None
if proc_id == 0:
    rng = np.random.randint(low = 1, high = 100000)
comm.barrier()

rng = comm.bcast(rng, root=0)

lmp.command('create_atoms 2 random %d %d void_region overlap 0.5 maxtry 50' % (n_h, rng))

lmp.command('dump mydump all custom 1000 %s/test_helium.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))

lmp_class.cg_min(lmp)

lmp_class.run_MD(lmp, 600, 1e-3, 10000)

lmp_class.cg_min(lmp)
