import sys
import os
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder):

    neb_input_file = '''
    units metal 
    atom_style atomic
    atom_modify map array sort 0 0.0
    read_data %s
    mass 1 183.84
    mass 2 1.00784
    mass 3 4.002602
    pair_style eam/%s
    pair_coeff * * %s W H He
    thermo 10
    run 0
    fix 1 all neb 1
    timestep 1e-3
    min_style quickmin
    thermo 100 
    variable i equal part
    neb 10e-15 10e-18 50000 50000 1000 final %s
    write_dump all custom %s/neb.$i.atom id type x y z ''' %  (init_path, pottype, potfile, final_path, neb_image_folder)

    return neb_input_file

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

print(proc_id, n_procs)

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'neb_datafiles'

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))
    os.mkdir(os.path.join(output_folder,'Neb_Image_Folder'))

potfile = init_dict['potfile']

pottype = potfile.split('.')[-1]

neb_image_folder = os.path.join(output_folder,'Neb_Image_Folder')

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp_class.perfect_crystal()

cmd = lmp_class.init_from_datafile(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'))

lmp = lammps(comm=comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(cmd)

lmp.command('create_atoms 3 single %f %f %f units box' % (2.25*3.14221, 2.5*3.14221, 2*3.14221))

lmp_class.cg_min(lmp)

lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'tet_1.data'))

lmp.command('write_dump all custom %s id x y z'  % os.path.join(output_folder, 'Atom_Files', 'tet_1.atom'))

lmp.close()


lmp = lammps(comm=comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(cmd)

lmp.command('create_atoms 3 single %f %f %f units box' % (2*3.14221, 2.5*3.14221, 2.25*3.14221))

lmp_class.cg_min(lmp)

lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'tet_2.data'))

lmp.command('write_dump all custom %s id x y z'  % os.path.join(output_folder, 'Atom_Files', 'tet_2.atom'))

lmp.close()

init_path = os.path.join(output_folder, 'Atom_Files', 'tet_1.atom')

final_path = os.path.join(output_folder, 'Atom_Files', 'tet_2.atom')

with open(final_path, 'r') as file:
    lines = file.readlines()

with open(final_path, 'w') as file:
    file.write(lines[3])
    file.writelines(lines[9:])

neb_txt = gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder)

with open('tet_tet.neb', 'w') as file:
    file.write(neb_txt)
