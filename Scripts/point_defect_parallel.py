import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

# comm = 0

# proc_id = 0

# n_procs = 1
from mpi4py import MPI

comm_gbl = MPI.COMM_WORLD

proc_id_gbl = comm_gbl.Get_rank()

n_procs = comm_gbl.Get_size()

color = proc_id_gbl // 4

comm = comm_gbl.Split(color, proc_id_gbl) 

proc_id = comm.Get_rank()

init_dict = {}

with open(sys.argv[1], 'r') as file:
    init_dict = json.load(file)

output_folder = 'lcl_files_%d' % color

init_dict['output_folder'] = output_folder

if proc_id == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))

comm.barrier()

lmp = LammpsParentClass(init_dict, comm, proc_id)

lmp.perfect_crystal()

x = np.arange(3)
y = np.arange(5)
z = np.arange(10)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).reshape(-1, 3)

ef_arr = np.zeros((len(points), ))

rvol_arr = np.zeros((len(points), ))

data = [[0, 0, 0, 0, 0]]

for _iter in range(3):
    for i, pt in enumerate(points[1:]):
        
        defect_centre = ( lmp.alattice*lmp.size/2 )*np.ones((3,))
        
        if pt[0] == 2:
            defect_centre = np.vstack([defect_centre, ( lmp.alattice* (lmp.size/2 - 0.5) )*np.ones((3,))])

        target_species = 2
        action = 1

        if pt[1] == 0:
            target_species = 3

            if pt[2] == 0:
                target_species = 1
                action = -1

        init_pt = np.copy(pt)

        if target_species == 1:
            init_pt[target_species - 1] += action
        else:
            init_pt[target_species - 1] -= action


        init_pt = np.clip(init_pt, a_min=0, a_max=np.inf).astype(int)

        input_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, init_pt[0], init_pt[1], init_pt[2])
        
        output_filepath = '%s/Data_Files/V%dH%dHe%d.data' % (output_folder, pt[0], pt[1], pt[2])
        
        ef, rvol = lmp.add_defect(input_filepath, output_filepath, target_species, action, defect_centre, minimizer='random',run_MD=True)
        
        data.append([pt[0], pt[1], pt[2], ef, rvol])

        if proc_id == 0:
            print(input_filepath, defect_centre, pt, ef, rvol)

    data = np.array(data)

if proc_id == 0:
    np.savetxt('%s-out.%d.txt' % (sys.argv[1].split('.')[0], color), data)