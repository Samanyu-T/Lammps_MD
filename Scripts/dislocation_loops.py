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

init_dict['orientx'] = [1, 1, -1]

init_dict['orienty'] = [-1, 1, 0]

init_dict['orientz'] = [1, 1, 2]


# init_dict['orientx'] = [1, 0, 0,]

# init_dict['orienty'] = [0, 1, 0]

# init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 10

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.he'

init_dict['pottype'] = 'he'

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_box()) 

a = 7

b = 5

n = 3

t = 2

d = 2

pts_arr = np.empty((0,2))

for i in range(t + 1):

    a_arr = np.arange(n, 2*a - n + 1).reshape(-1, 1)

    b_arr = np.arange(n, 2*b - n + 1).reshape(-1, 1)

    pts = np.vstack([
        np.hstack([a_arr ,np.zeros(a_arr.shape)]),
        np.hstack([2*a*np.ones(b_arr.shape) , b_arr]),
        np.hstack([np.flip(a_arr) ,2*b*np.ones(a_arr.shape)]),
        np.hstack([np.zeros(b_arr.shape) ,np.flip(b_arr)]),
    ])

    if n > 1:
        diag = np.arange(n + 1)[1:-1].reshape(-1, 1)

        diag_pts = np.vstack([
            np.hstack([diag ,diag + 2*b - n]),
            np.hstack([2*a - n + diag , diag]),
            np.hstack([diag ,np.flip(diag)]),
            np.hstack([2*a -n + diag , 2*b - n + np.flip(diag)]),
        ])
        
        print(diag_pts)

        pts = np.vstack([pts, diag_pts])

    pts += np.array([0.5 + i,0 + i])
    pts *= 0.5

    pts = np.unique(pts, axis=0)
    
    pts_arr = np.vstack([pts_arr, pts])

    a -= 1

    b -= 1

pts_arr = np.unique(pts_arr, axis=0)

# plt.scatter(pts_arr[:,0], pts_arr[:,1])
# plt.show()
# print(pts_arr)

for i in range(d):
    for _pt in pts_arr:
        lmp.command('create_atoms 1 single %f %f %f units lattice' % (_pt[0], _pt[1], i*0.5))

lmp.command('write_dump all custom %s/test.atom id type x y z' % output_folder)

lmp_class.cg_min(lmp, conv=10000)