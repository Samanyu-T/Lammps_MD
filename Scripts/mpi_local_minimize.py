import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import FS_Fitting
import Handle_PotFiles_FS
import time
import json, glob, shutil
from scipy.optimize import minimize
from mpi4py import MPI

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
        files_to_copy.extend(glob.glob('%s/V3H0He1.*.txt' % data_dir))

    if he_he:
        # files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He2.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He3.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H0He4.*.txt' % data_dir))

    if h_he:
        # files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He0.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He3.*.txt' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
    

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, proc_id)

# comm = 0

# proc_id = 0

# n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/init.eam.fs')

n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 2
n_knots['He-H p'] = 2
n_knots['He-He p'] = 2
n_knots['W-He'] = 4
n_knots['He-He'] = 4
n_knots['H-He'] = 4

x1 = np.array([5.46, 0.91861628,  1.27351374, 2, 1, 2,  0.78466961, 2, 3e-4, 
              -1.89051467,  3.00833016,  1.85408476, -0.64971704,  0.63928971, -0.32182122,
              -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
              -0.1294734, 0.1495497,  -0.36530064, -0.03049166,  0.03071201, -0.06344354]).reshape(1, -1)

x2 = np.array([5.46, 0.28674766 , 0.22296933,  0.782321 ,   0.66957067, 28.30868662 , 0.10724439, 2, 3e-4,
                -2.16033878 , 3.34641422,  0.72020167 ,-0.67347082,  0.98436323, -0.80477617,
                -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
                -0.19861687 , 0.38362381, -0.53700151, -0.03548229,  0.04469245, -0.05161365]).reshape(1, -1)

x3 = np.array([5.46, 0.44923357,  0.25870337,  1.47446042,  0.53508436 , 1.75218746,  0.37262526, 2, 3e-4,
                -2.04223323,  3.15077119 , 1.82794484, -0.67536188 , 0.82282924, -0.47672833,
                -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
                -0.17087058,  0.28199124 ,-0.42701823 ,-0.03128522 , 0.0346663 , -0.07079101]).reshape(1, -1)

cov1 = np.array([2.5e-1, 2.36957394e-01, 1.15217341e-01, 2.20532689e-01, 2.73011654e-01, 1.29271219e-01,1.94412517e-01, 1e-1, 1e-1,
                 2.14840428e-02, 1.01185895e-01, 5.08868551e-01, 2.23807029e-03, 2.64009290e-02, 9.57105848e-02,
                 4.30405244e-04, 4.35567428e-03, 9.42275179e-03, 2.56976122e-05, 2.15198004e-04 , 7.84448617e-03,
                 2.30405244e-03, 1.35567428e-02, 9.42275179e-02, 2.56976122e-05, 2.15198004e-04 , 7.84448617e-03,])

cov_full = np.array([np.diag(cov1), np.diag(cov1), np.diag(cov1)])

mean_full = np.vstack([x1, x2, x3])

mean = mean_full[proc_id % mean_full.shape[0]]

cov = cov_full[proc_id % mean_full.shape[0]]

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

save_folder = os.path.join(param_dict['save_dir'], 'Local_Minimizer')

if proc_id == 0:
    copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

    if not os.path.exists(save_folder) and proc_id == 0:
        os.mkdir(save_folder)

comm.Barrier()

eam_fit = FS_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])


data_ref = np.loadtxt('dft_yang.txt')

for i in range(20):

    x_init = np.random.multivariate_normal(mean=mean, cov=cov)

    FS_Fitting.simplex(n_knots, comm, proc_id, x_init, 700, param_dict['work_dir'], save_folder)
