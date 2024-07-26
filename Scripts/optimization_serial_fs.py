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
        files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))

    if h_he:
        files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He0.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He1.*.txt' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/init.eam.fs')


n_knots = {}
n_knots['He F'] = 0
n_knots['H-He p'] = 0
n_knots['He-W p'] = 2
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 4

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = FS_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

sample2 = eam_fit.gen_rand()

sample = np.loadtxt('sample.txt') 
# -2.07867732  4.00598855 -2.6092615  -0.66293926  0.87068177 -1.12676143
# -2.01678669  2.98128083  2.54008138 -0.74283541  0.93099271 -0.47204214
# -1.89051467  3.00833016  1.85408476 -0.64971704  0.63928971 -0.32182122
# 5.30600542  1.99987348 0.6 -1.93906141  2.7796295   1.5790798  -0.52774223  0.56313272 -0.4323598  -1.04069087  0.48249025 -0.20081857 -0.00879394 -0.15072352  1.14083218
# sample = np.array([1e-4])
# sample += 1e-2*np.random.random(sample.shape)
print(sample2.shape, sample.shape)
data_ref = np.loadtxt('dft_update.txt')

t1 = time.perf_counter()

FS_Fitting_Serial.simplex(n_knots, comm, proc_id, sample, 1000, param_dict['work_dir'], param_dict['save_dir'])

t2 = time.perf_counter()
print(t2 - t1)