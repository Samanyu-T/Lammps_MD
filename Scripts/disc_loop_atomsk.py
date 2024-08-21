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

init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.he'

init_dict['pottype'] = 'he'

output_folder = 'Dislocation_Loops'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

# lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

# lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_loop_111.lmp')) 

# fix_length = 10

# lmp.command('region centre block %f %f %f %f %f %f side in units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('yhi') - fix_length ,
#                                                   0, lmp.get_thermo('zhi')  ))

# lmp.command('group fix_atoms_0 region centre')

# lmp.command('fix freeze_0 fix_atoms_0 setforce 0.0 0.0 0.0')

# lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)


# lmp.command('unfix freeze_0')


# lmp.command('region out_centre block %f %f %f %f %f %f side out units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
#                                                   fix_length, lmp.get_thermo('yhi') - fix_length ,
#                                                   0, lmp.get_thermo('zhi') ))

# lmp.command('group fix_atoms_1 region centre')

# lmp.command('fix freeze_1 fix_atoms_1 setforce 0.0 0.0 0.0')

# lmp_class.cg_min(lmp, conv=10000, fix_aniso=False)

# pe_loop_0 = lmp.get_thermo('pe')

# lmp.command('create_atoms 3 single 33 33 1 units box')

# lmp_class.cg_min(lmp, conv=10000)

# pe_loop_1 = lmp.get_thermo('pe')

# xyz = np.array(lmp.gather_atoms('x', 1, 3))

# xhi = lmp.get_thermo('xhi') - 300

# yhi = lmp.get_thermo('yhi') - 300

# xyz_loop_he = xyz[-3:]

# xyz_loop_he[0] -= xhi * 0.501
# xyz_loop_he[1] -= yhi * 0.501

# lmp.command('write_dump all custom %s/test.0.atom id type x y z' % output_folder)

# lmp.close()


lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_edge_111.lmp')) 


fix_length = 10

lmp.command('region centre block %f %f %f %f %f %f side in units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('yhi') - fix_length ,
                                                  0, lmp.get_thermo('zhi') ))

lmp.command('group fix_atoms_0 region centre')

lmp.command('fix freeze_0 fix_atoms_0 setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=True)


lmp.command('unfix freeze_0')


lmp.command('region out_centre block %f %f %f %f %f %f side out units box' % (fix_length, lmp.get_thermo('xhi') - fix_length ,
                                                  fix_length, lmp.get_thermo('yhi') - fix_length ,
                                                  0, lmp.get_thermo('zhi')))

lmp.command('group fix_atoms_1 region centre')

lmp.command('fix freeze_1 fix_atoms_1 setforce 0.0 0.0 0.0')

lmp_class.cg_min(lmp, conv=10000, fix_aniso=False)

pe_edge_0 = lmp.get_thermo('pe')

lmp.command('create_atoms 3 single 33 33 1 units box')

lmp_class.cg_min(lmp, conv=10000)

pe_edge_1 = lmp.get_thermo('pe')

xyz = np.array(lmp.gather_atoms('x', 1, 3))

xhi = lmp.get_thermo('xhi') - 300

yhi = lmp.get_thermo('yhi') - 300

xyz_edge_he = xyz[-3:]

xyz_edge_he[0] -= xhi * 0.51
xyz_edge_he[1] -= yhi * 0.51

lmp.command('write_dump all custom %s/test.1.atom id type x y z' % output_folder)

lmp.close()


lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Dislocations/W_screw_111.lmp')) 

fix_length = 20

lmp_class.cg_min(lmp, conv=10000, fix_aniso=False)

pe_screw_0 = lmp.get_thermo('pe')

lmp.command('create_atoms 3 single 32 33 1 units box')

lmp_class.cg_min(lmp, conv=10000)

pe_screw_1 = lmp.get_thermo('pe')

xyz = np.array(lmp.gather_atoms('x', 1, 3))

xhi = lmp.get_thermo('xhi') - 300

yhi = lmp.get_thermo('yhi') - 300

xyz_screw_he = xyz[-3:]

xyz_screw_he[0] -= xhi * 0.51
xyz_screw_he[1] -= yhi * 0.51

lmp.command('write_dump all custom %s/test.2.atom id type x y z' % output_folder)

lmp.close()

# print(pe_loop_0, pe_loop_1, xyz_loop_he)

C11 = 3.229
C12 = 1.224
C44 = 0.888

C_voigt = np.array([[C11, C12, C12, 0, 0, 0],
              [C12, C11, C12, 0, 0, 0],
              [C12, C12, C11, 0, 0, 0],
              [0, 0, 0, C44, 0, 0],
              [0, 0, 0, 0, C44, 0],
              [0, 0, 0, 0, 0, C44]])

Cinv = np.linalg.inv(C_voigt)

G = 1.020
b = 2.721233684
v = 0.2819650067
t = G / (2* np.pi * (1 - v))
a = 3.145
rvol = 0.32662381279305125

strain = np.array([2.60513267e-04, 4.21136936e-05, 1.73493355e-04, 0, 0, 0])

x = xyz_edge_he[0] 
y = xyz_edge_he[1] 
xx = (b / (2 * np.pi)) * ( - y / (x**2 + y**2) + (y**3 - 2*x**2*y)/( (2* (1-v) * (x**2 + y**2)**2 ))) 
yy = (b / (2 * np.pi)) * ((y**3 + x**2*y - 2*x*y**2)/( (2* (1-v) * (x**2 + y**2)**2 ))) 
zz = 0

stress = np.array([xx, yy, zz, 0, 0, 0])

print( (C11 + 2*C12) * np.sum(stress) * rvol * (a**3/2) / 3,  stress)

xz = - ( (G * b) / (2* np.pi)) * xyz_edge_he[1] / (xyz_edge_he[0] **2 + xyz_edge_he[1] **2)
yz = - ( (G * b) / (xyz_edge_he[0] )) * xyz_edge_he[1] / (xyz_edge_he[0] **2 + xyz_edge_he[1] **2)
 
stress = np.array([0, 0, 0, xz, yz, 0])

print( (C11 + 2*C12) * np.sum(stress) * rvol * (a**3/2) ,  stress)

print(pe_edge_0, pe_edge_1, xyz_edge_he)

print(pe_screw_0, pe_screw_1, xyz_screw_he)