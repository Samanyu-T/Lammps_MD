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

output_folder = 'H_Retention'
save_folder = os.path.join(output_folder,'Iteration_0')

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(save_folder)

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

cv = 1e-2
ch = 1e-2
che = 1e-2

nv = 2 * cv * lmp_class.size ** 3
nh = 2 * ch * lmp_class.size ** 3
nhe = 2 * che * lmp_class.size ** 3


lmp = lammps()#cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

temp = 300 

lmp.commands_list(lmp_class.init_from_box()) 

rng = np.random.randint(low = 1, high = 100000)

lmp.command('delete_atoms random count %d no all NULL %d' % (nv, rng))    

rng = np.random.randint(low = 1, high = 100000)

lmp.command('create_atoms 2 random %d %d NULL overlap 0.5 maxtry 100' % (nh, rng))    

rng = np.random.randint(low = 1, high = 100000)

lmp.command('create_atoms 3 random %d %d NULL overlap 0.5 maxtry 100' % (nhe, rng))    

rng = np.random.randint(low = 1, high = 100000)

lmp.command('group mobile type 2 3')

lmp.command('fix fix_temp all nve')

lmp.command('velocity all create %f %d rot no dist gaussian' % (2*temp, rng))

lmp.command('dump mydump all custom 1000 %s/cascade.*.atom id type x y z'  % save_folder)

lmp.command('timestep 1e-3')

lmp.command('run %d' % (5e6))