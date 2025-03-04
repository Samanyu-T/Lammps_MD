import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import He_Fitting
import Handle_PotFiles_He
import time
import json, glob, shutil
import matplotlib.pyplot as plt
from mpi4py import MPI

def copy_files(w_he, he_he, h_he, work_dir, data_dir):
    
    data_files_folder = os.path.join(work_dir, 'Data_Files')

    if proc_id == 0:
        if os.path.exists(data_files_folder):
            shutil.rmtree(data_files_folder)

        os.mkdir(data_files_folder)

    files_to_copy = []
    
    files_to_copy.extend(glob.glob('%s/V*H0He0.0.txt' % data_dir))

    if w_he:
        files_to_copy.extend(glob.glob('%s/V0H0He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V3H0He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He1.0.txt' % data_dir))

    if he_he:
        files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H0He3.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H0He4.*.txt' % data_dir))

    if h_he:
        # files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He0.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He1.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He3.*.txt' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    
comm = MPI.COMM_WORLD

# proc_id = 0

# n_procs = 1

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/final.eam.he')

n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 0
n_knots['He-W p'] = 0
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 0
n_knots['W-He p'] = 0

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

sample2 = eam_fit.gen_rand()

sample = np.loadtxt('sample.txt') 

eam_fit.sample_to_file(sample)

whe = eam_fit.pot_lammps['W-He'][1:]

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]

phi = whe/r

plt.plot(r[400:], phi[400:])

plt.show()

plt.plot(r, eam_fit.pot_lammps['He-W p'][1:])

plt.show()

print(sample2.shape, sample.shape)

data_ref = np.loadtxt('dft_yang.txt')

t1 = time.perf_counter()

He_Fitting.simplex(n_knots, comm, proc_id, sample , 1000, param_dict['work_dir'], param_dict['save_dir'])

t2 = time.perf_counter()
print(t2 - t1)

# -1.12629766  1.00451879 50.83365775  0.49178287  3.92100575 -1.08399913  1.38391886  9.5296594  -0.16774509  0.43470896 -0.78895374