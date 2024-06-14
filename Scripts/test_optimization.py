import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import EAM_Fitting
import Handle_PotFiles
import time
from mpi4py import MPI
import json, glob, shutil

def copy_files(w_he, he_he, h_he, work_dir, data_dir):
    
    data_files_folder = os.path.join(work_dir, 'Data_Files')

    if os.path.exists(data_files_folder):
        shutil.rmtree(data_files_folder)

    os.mkdir(data_files_folder)

    files_to_copy = []
    
    files_to_copy.extend(glob.glob('%s/V*H0He0.0.data' % data_dir))

    if w_he:
        files_to_copy.extend(glob.glob('%s/V0H0He1.*.data' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He1.0.data' % data_dir))

    if he_he:
        files_to_copy.extend(glob.glob('%s/V*H0He*.*.data' % data_dir))

    if h_he:
        files_to_copy.extend(glob.glob('%s/V*H*He*.*.data' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    
comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()


os.chdir('/Users/cd8607/Documents/Lammps_MD')


pot, potlines, pot_params = Handle_PotFiles.read_pot('git_folder/Potentials/test.eam.alloy')

n_knots = {}
n_knots['He_F'] = 2
n_knots['He_p'] = 3
n_knots['W-He'] = 3
n_knots['He-He'] = 0
n_knots['H-He'] = 0


with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, False, False, param_dict['work_dir'], param_dict['data_dir'])


eam_fit = EAM_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

sample = np.array([6.467249380859618313e+00, 2.005920277681737751e+00, -4.223553075097251863e+00, 4.832841278475297209e-01,
                   -1.333137585250958379e+00, 1.554035936040987220e+00, 8.629746307382948345e-01, -4.633698171737774985e+00,
                    8.567061272702570562e+00])

data_ref = np.loadtxt('dft_data_new.txt')

t1 = time.perf_counter()

print(EAM_Fitting.loss_func(sample,data_ref,eam_fit))

t2 = time.perf_counter()

print(t2 - t1)