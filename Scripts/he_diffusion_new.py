import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
from Lammps_Classes_Serial import LammpsParentClass

comm = 0

proc_id = 0

n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'He_Diffusion_Serial'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Iteration_0'))

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

temp = 300 

init_file = 'Lammps_Files_7x7/Data_Files/V0H0He1.data'

lmp.commands_list(lmp_class.init_from_datafile(init_file)) 

rng = np.random.randint(low = 1, high = 100000)

lmp.command('group mobile type 2 3')

lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rng))

lmp.command('fix fix_temp all nve')

lmp.command('dump mydump mobile custom 1000 %s/Iteration_0/cascade.*.atom id type x y z'  % output_folder)

lmp.command('timestep 1e-3')

lmp.command('run %d' % (5e6))