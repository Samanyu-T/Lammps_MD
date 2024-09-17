import sys
import os
import json
import numpy as np
from mpi4py import MPI
# import matplotlib.pyplot as plt
from lammps import lammps

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder, natoms):

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
# partition yes 1 fix freeze all setforce 0.0 0.0 0.0
# partition yes 7 fix freeze all setforce 0.0 0.0 0.0
fix 1 all neb 10000
timestep 1e-4
min_style quickmin
thermo 100 
variable i equal part
neb 10e-15 10e-18 50000 50000 10000 final %s
write_dump all custom %s/neb.$i.atom id type x y z
variable he_z equal z[%d]
print "He_z $(v_he_z:%s)" ''' %  (init_path, pottype, potfile, final_path, neb_image_folder, natoms,'%.6f')

    return neb_input_file

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

# comm = 0 
# proc_id = 0
# n_procs = 1

print(proc_id, n_procs)

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

init_dict['size'] = 7

init_dict['surface'] = 10

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

output_folder = 'Dislocation_Loops'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))
    os.mkdir(os.path.join(output_folder,'Neb_Image_Folder'))

init_dict['output_folder'] = output_folder

potfile = init_dict['potfile']

pottype = potfile.split('.')[-1]

neb_image_folder = os.path.join(output_folder,'Neb_Image_Folder')

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('%s/vac_loop_he.data' % os.path.join(output_folder,'Data_Files')))

lmp_class.cg_min(lmp)
pe_1 = lmp.get_thermo('pe')

print(pe_1)


init_path = os.path.join(output_folder, 'Data_Files', 'vacloop_1.data')

final_path = os.path.join(output_folder, 'Atom_Files', 'vacloop_2.atom')


lmp.command('write_data %s' % init_path)

lmp.command('write_dump all custom %s id x y z'  % os.path.join(output_folder, 'Atom_Files', 'vacloop_1.atom'))

lmp.close()


lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('%s/vac_loop_he.data' % os.path.join(output_folder,'Data_Files')))
lmp.command('group ghelium type 3')
lmp.command('displace_atoms ghelium move 0 0 3 units lattice')
lmp_class.cg_min(lmp)

# lmp_class.run_MD(lmp, temp=600, timestep=1e-3, N_steps= 10000)

lmp_class.cg_min(lmp)

pe_1 = lmp.get_thermo('pe')

print(pe_1)

lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'vacloop_2.data'))

lmp.command('write_dump all custom %s id x y z'  % final_path)

natoms = lmp.get_natoms()

lmp.close()

with open(final_path, 'r') as file:
    lines = file.readlines()

with open(final_path, 'w') as file:
    file.write(lines[3])
    file.writelines(lines[9:])

neb_txt = gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder, natoms)

with open('vac_loop.neb', 'w') as file:
    file.write(neb_txt)
