import sys
import os
import json
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes_Serial import LammpsParentClass

comm = 0

proc_id = 0

n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'EAM_Fit_Files'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 4

init_dict['surface'] = 0

init_dict['potfile'] = 'git_folder/Potentials/test.0.eam.alloy'

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = LammpsParentClass(init_dict, comm, proc_id)

lmp.perfect_crystal()

x = np.arange(3)
y = np.arange(5)
z = np.arange(5)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).reshape(-1, 3)

ef_arr = np.zeros((len(points), ))

rvol_arr = np.zeros((len(points), ))

dft_data = np.loadtxt('dft_data_new.txt')

for i, pt in enumerate(dft_data[5:, :3]):
    
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

    # print(input_filepath, output_filepath, init_pt, pt, action)
    
    ef, rvol = lmp.add_defect(input_filepath, output_filepath, target_species, action, defect_centre)
    
    print(output_filepath, ef, rvol)
