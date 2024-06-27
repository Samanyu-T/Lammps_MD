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

lmp.commands_list(lmp_class.init_from_datafile('Atomsk_Files/Grain_Boundaries/polycrystal.lmp')) 

lmp_class.cg_min(lmp, conv=100000, fix_aniso=True)

pe0 = lmp.get_thermo('pe')

xl = lmp.get_thermo('lx')
yl = lmp.get_thermo('ly')
zl = lmp.get_thermo('lz')

lmp.command('create_atoms 3 single %f %f %f' % (xl/2, yl/2, zl/2))

lmp_class.cg_min(lmp, conv=100000, fix_aniso=True)

pe1 = lmp.get_thermo('pe')

print(pe0, pe1, 6.13)
lmp.command('write_dump all custom %s/poly.atom id type x y z' % output_folder)

lmp.close()
