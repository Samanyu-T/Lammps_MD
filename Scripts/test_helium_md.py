import sys
import os
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes_Serial import LammpsParentClass

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm = 0

proc_id = 0

n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Test_Helium'


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))



init_dict['size'] = 5

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

# init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.fs'

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp_class.perfect_crystal()

cmd = lmp_class.init_from_datafile(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'))


lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(cmd)

n_vac = 0

n_he = 10

for i in range(n_vac):
    rand = np.random.randint(low=1, high=lmp.get_natoms())
    lmp.command('group del id %d' % rand)
    lmp.command('delete_atoms group del compress yes')

rng = np.random.randint(low=1, high=1e6)
lmp.command('create_atoms 3 random %d %d NULL overlap 1.0 maxtry 50' % (n_he, rng))

lmp.command('dump mydump all custom 1000 %s/test_helium.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))

lmp_class.run_MD(lmp, 500, 1e-3, 1000000)

lmp_class.cg_min(lmp)

