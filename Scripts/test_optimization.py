import numpy as np
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

sample = np.array([1.130350965490571546e+01, 2.752972140971631898e+00, -2.072280818241628353e+00,
                   7.097202368747449475e-01, -1.193421854629756140e+00, -3.655969179503565925e-01,
                     1.360741995181738329e+00, -3.153228806122611694e+00, -1.194795420738169689e+00])

data_ref = np.loadtxt('dft_data_new.txt')

t1 = time.perf_counter()

print(EAM_Fitting.loss_func(sample,data_ref,eam_fit))

t2 = time.perf_counter()

print(t2 - t1)