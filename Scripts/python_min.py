import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import FS_Fitting
import time
import json, glob, shutil
from mpi4py import MPI
from scipy import optimize
import Handle_PotFiles_FS

def copy_files(w_he, he_he, h_he, work_dir, data_dir):
    
    data_files_folder = os.path.join(work_dir, 'Data_Files')

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
        # files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))
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

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, proc_id)

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

max_time = param_dict['max_time']
work_dir = param_dict['work_dir']
save_folder = param_dict['save_dir']
data_dir = param_dict['data_dir']

if (not os.path.exists(save_folder)) and proc_id == 0:
    os.mkdir(save_folder)

if (not os.path.exists(work_dir)) and proc_id == 0:
    os.mkdir(work_dir)

if (not os.path.exists(os.path.join(work_dir, 'Potentials'))) and proc_id == 0:
    os.mkdir(os.path.join(work_dir, 'Potentials'))

comm.Barrier()

n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 2
n_knots['He-H p'] = 2
n_knots['He-He p'] = 2
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 0
n_knots['W-He p'] = 3
 
if proc_id == 0:
    copy_files(True, True, True, work_dir, data_dir)

comm.barrier()

sample_folder = ''

if proc_id == 0:
    print('Start Genetic Algorithm \n')
    sys.stdout.flush()  

rsamples_folder = os.path.join(save_folder, 'Global_Min') 

if not os.path.exists(rsamples_folder) and proc_id == 0:
    os.mkdir(rsamples_folder)
    
comm.Barrier()  

bounds = (
            (1, 10), (0, 1),
            (1, 100), (1, 10),
            (1, 100), (1, 10),
            (1, 100), (1, 10),
            (1, 100), (1, 10),
            (-3, -0.5), (1, 5), (-2, 5), (-1, 0.2), (-1, 2), (-3, 3),
            (0.2, 1), (-1, 0), (-0.1, 0.1), (0, 0.1), (-0.2, 0), (-0.1, 0.1)
            )       

x0 = np.array([ 1.67840e+00,  6.53800e-01,  10, 5, 7.15027e+01,  7.24880e+00, 10, 5, 10, 5,
       -8.02100e-01,  1.00690e+00,  3.70660e+00, -3.12400e-01,
        5.89600e-01, -4.94000e-01,  6.01300e-01, -1.97200e-01,
       -2.91000e-02,  1.92000e-02, -1.00000e-04, -6.62000e-02])

data_ref = np.loadtxt('dft_yang.txt')


pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/init.eam.fs')

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = FS_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

loss = np.empty((0))
samples = np.empty((0, 16))


for i in range(112):
    _loss = np.loadtxt('genetic/Loss_%d.txt' % i)
    _samples = np.loadtxt('genetic/Samples_%d.txt' % i)

    loss = np.hstack([loss, _loss])
    samples = np.vstack([samples, _samples])

idx = np.argsort(loss)
loss = loss[idx]
samples = samples[idx]

x0 = samples[:1000]

_samples = np.loadtxt('gmm/GMM_3/Filtered_Samples.txt')

x_init = np.vstack([x0, _samples[:100]])


test = np.hstack([np.clip(100 * np.random.rand(len(x_init), 1), a_max=100,a_min=1), 
                  np.clip(10 * np.random.rand(len(x_init), 1), a_max=10,a_min=1)])

x_init = np.hstack([x_init[:, :2], test, x_init[:, 2:4], test, test, x_init[:,4:]])

bounds_arr = np.array(bounds)

x_filtered = []

for _x in x_init:
    for i, __x in enumerate(_x):
        add = True
        if not (bounds_arr[i,0] <= __x <= bounds_arr[i,1]):
            add = False
            break
    if add:
        x_filtered.append(_x)

x_filtered = np.array(x_filtered)

optimize.differential_evolution(FS_Fitting.loss_func, bounds, args=(data_ref, eam_fit, True, True, rsamples_folder),
                                init=x_filtered, mutation=1, recombination=0.5, polish=True)