import sys
import os
import json
import numpy as np
from mpi4py import MPI
import pandas as pd
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def create_df(xyz, ef, rvol):

    data = np.column_stack((xyz[:,0].flatten(),
                            xyz[:,1].flatten(),
                            xyz[:,2].flatten(),
                            ef.flatten(),
                            rvol.flatten()))  
    
    sorted_indices = np.lexsort((data[:, 2], data[:, 1], data[:, 0]))

    data = data[sorted_indices]

    return data    

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'EAM_Fit_Files'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

# init_dict['potfile'] = 'git_folder/Potentials/WHHe_test.eam.alloy'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = LammpsParentClass(init_dict, comm, proc_id)
lmp.perfect_crystal()

dft_data = np.loadtxt('dft_data_new.txt')

defect_centre = ( lmp.alattice*lmp.size/2 )*np.ones((3,))

pos_arr = defect_centre + lmp.alattice*np.array([[0.25, 0.5, 0], [0.15, 0.5, 0], [0.375, 0.375, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25]])

for vac in range(3):
    for i, _pos in enumerate(pos_arr):

        lmp.create_atoms_given_pos('EAM_Fit_Files/Data_Files/V%dH0He0.data' % vac, 'EAM_Fit_Files/Data_Files/V%dH0He1.%d.data' % (vac, i),
                                        [3], [_pos], run_MD=False, run_min=False)


for i in range(5):
    shutil.copy('EAM_Fit_Files/Data_Files/V0H0He0.data', 'EAM_Fit_Files/Data_Files/V0H0He0.%d.data' % i)
    shutil.copy('EAM_Fit_Files/Atom_Files/V0H0He0.atom', 'EAM_Fit_Files/Atom_Files/V0H0He0.%d.atom' % i)
    shutil.copy('EAM_Fit_Files/Data_Files/V1H0He0.data', 'EAM_Fit_Files/Data_Files/V1H0He0.%d.data' % i)
    shutil.copy('EAM_Fit_Files/Data_Files/V2H0He0.data', 'EAM_Fit_Files/Data_Files/V2H0He0.%d.data' % i)

lmp.conv = 1000

for i, pt in enumerate(dft_data[:, :3]):

    for img in range(5):

        if (pt == np.array([0,0,1])).all():
            continue
        
        if (pt == np.array([0,0,0])).all():
            continue

        defect_centre = ( lmp.alattice*lmp.size/2 )*np.ones((3,))
        
        if pt[0] == 2:
            defect_centre = np.vstack([defect_centre, ( lmp.alattice* (lmp.size/2 + 0.5) )*np.ones((3,))])

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

        input_filepath = '%s/Data_Files/V%dH%dHe%d.%d.data' % (output_folder, init_pt[0], init_pt[1], init_pt[2], img)
        
        output_filepath = '%s/Data_Files/V%dH%dHe%d.%d.data' % (output_folder, pt[0], pt[1], pt[2], img)

        # print(input_filepath, output_filepath, init_pt, pt, action)
        
        ef, rvol = lmp.add_defect(input_filepath, output_filepath, target_species, action, defect_centre)


        print(output_filepath, ef, rvol)