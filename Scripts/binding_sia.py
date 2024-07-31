import sys
import os
import json
import numpy as np
# from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps

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

output_folder = 'SIA_Binding'


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

def surface_binding(lmp, depth):

    pe_arr = np.zeros((10,))

    for i, _d in enumerate(depth):
        pe, _ = lmp.add_defect(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'), 
                               os.path.join(output_folder, 'Data_Files', 'test.data'), 3, 1, np.array([0, 0, _d]))
        
        print(_d, pe - lmp.pe_perfect)
        
        pe_arr[i] = pe

    pe_arr -= lmp.pe_perfect

    return lmp, pe_arr



init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.fs'

init_dict['pottype'] = 'fs'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp_class.perfect_crystal()

cmd = lmp_class.init_from_datafile(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'))

lmp = lammps()

lmp.commands_list(cmd)

lmp.command('create_atoms 1 single %f %f %f units box' % (2.25*3.14221, 2.25*3.14221, 2.25*3.14221))

lmp_class.cg_min(lmp)

lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'sia.data'))

lmp.command('write_dump all custom %s id type x y z'  % os.path.join(output_folder, 'Atom_Files', 'sia.atom'))

lmp.close()

cmd = lmp_class.init_from_datafile(os.path.join(output_folder, 'Data_Files', 'sia.data'))

lmp = lammps()

lmp.commands_list(cmd)

lmp.command('create_atoms 1 single %f %f %f units box' % (2.75*3.14221, 2.75*3.14221, .752*3.14221))

lmp_class.cg_min(lmp)

pe = lmp.get_thermo('pe')

lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'sia.data'))

lmp.command('write_dump all custom %s id type x y z'  % os.path.join(output_folder, 'Atom_Files', 'sia.atom'))

lmp.close()

print(pe)