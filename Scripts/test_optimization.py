import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import EAM_Fitting_Serial
import Handle_PotFiles
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

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles.read_pot('git_folder/Potentials/beck.eam.alloy')


n_knots = {}
n_knots['He_F'] = 2
n_knots['He_p'] = 4
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 0


with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, False, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = EAM_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

sample2 = eam_fit.gen_rand()

sample = np.loadtxt('sample.txt')

print(sample2.shape, sample.shape)
data_ref = np.loadtxt('dft_update.txt')

t1 = time.perf_counter()

EAM_Fitting_Serial.simplex(n_knots, comm, proc_id, sample, 1000, param_dict['work_dir'], param_dict['save_dir'])

t2 = time.perf_counter()
print(t2 - t1)