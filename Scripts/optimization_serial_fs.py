import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import FS_Fitting_Serial
import Handle_PotFiles_FS
import time
import json, glob, shutil
import matplotlib.pyplot as plt

def copy_files(w_he, he_he, h_he, work_dir, data_dir):
    
    data_files_folder = os.path.join(work_dir, 'Data_Files')

    if os.path.exists(data_files_folder):
        shutil.rmtree(data_files_folder)

    os.mkdir(data_files_folder)

    files_to_copy = []
    
    files_to_copy.extend(glob.glob('%s/V*H0He0.0.txt' % data_dir))

    if w_he:
        files_to_copy.extend(glob.glob('%s/V0H0He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He1.0.txt' % data_dir))

    if he_he:
        # files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He2.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He3.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He4.*.txt' % data_dir))

    if h_he:
        # files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He0.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He1.*.txt' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/init.eam.fs')


n_knots = {}
n_knots['He F'] = 0
n_knots['H-He p'] = 2
n_knots['He-W p'] = 2
n_knots['He-H p'] = 2
n_knots['He-He p'] = 0
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 4

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = FS_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

zsample2 = eam_fit.gen_rand()

sample = np.loadtxt('sample.txt') 
#  1.15577835  1.37018979  1.14701264  0.91343181  2.26586415  1.0037183
#  -1.89165792  2.74925153  1.88329318 -0.63323941  0.63944221 -0.32335614
#  -0.10747245  0.13973368 -0.36959604 -0.02580016  0.0362517  -0.05010032
# sample = np.array([1e-4])
# sample += 1e-2*np.random.random(sample.shape)
# print(sample2.shape, sample.shape)

data_ref = np.loadtxt('dft_data_final.txt')

t1 = time.perf_counter()

FS_Fitting_Serial.simplex(n_knots, comm, proc_id, sample + 1e-2*np.random.randn(len(sample)), 1000, param_dict['work_dir'], param_dict['save_dir'])

t2 = time.perf_counter()
print(t2 - t1)