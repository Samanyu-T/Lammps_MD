import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes_Serial import LammpsParentClass
import matplotlib.pyplot as plt

comm = 0

proc_id = 0

n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Dislocation_Loops'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_loop_111.lmp')) 

fix_length = 20

lmp.command('region centre block %f %f %f %f %f %f side in units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('yhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('zhi') - fix_length ))

lmp.command('group fix_atoms_0 region centre')

lmp.command('fix freeze_0 fix_atoms_0 setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)


lmp.command('unfix freeze_0')


lmp.command('region out_centre block %f %f %f %f %f %f side out units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('yhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('zhi') - fix_length ))

lmp.command('group fix_atoms_1 region centre')

lmp.command('fix freeze_1 fix_atoms_1 setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)

pe0 = lmp.get_thermo('pe')

lmp.command('create_atoms 3 single 33 33 41 units box')

lmp_class.cg_min(lmp, conv=10000)
pe1 = lmp.get_thermo('pe')

lmp.command('write_dump all custom %s/test.0.atom id type x y z' % output_folder)

lmp.close()


lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_edge_111.lmp')) 

# lmp.command('region centre block %f %f %f %f %f %f side in units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('yhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('zhi') - fix_length ))

# lmp.command('group fix_atoms region centre')

# lmp.command('fix freeze fix_atoms setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)

# lmp.command('unfix freeze')

# lmp_class.cg_min(lmp, conv=10000)

lmp.command('write_dump all custom %s/test.1.atom id type x y z' % output_folder)

lmp.close()


lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_screw_111.lmp')) 

fix_length = 20

lmp.command('region centre block %f %f %f %f %f %f side in units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('yhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('zhi') - fix_length ))

# lmp.command('group fix_atoms_0 region centre')

# lmp.command('fix freeze_0 fix_atoms_0 setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)


# lmp.command('unfix freeze_0')


# lmp.command('region out_centre block %f %f %f %f %f %f side out units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('yhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('zhi') - fix_length ))

# lmp.command('group fix_atoms_1 region centre')

# lmp.command('fix freeze_1 fix_atoms_1 setforce 0.0 0.0 0.0')

# lmp_class.cg_min(lmp, conv=10000)

lmp.command('write_dump all custom %s/test.2.atom id type x y z' % output_folder)

lmp.close()
print(pe1, pe0, 6.13)