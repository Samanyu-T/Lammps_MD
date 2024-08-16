import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import He_Fitting
import time
import json, glob, shutil
from mpi4py import MPI
from scipy import optimize
import Handle_PotFiles_He

# x0 = np.array([ 1.67840e+00,  6.53800e-01,  10, 5, 7.15027e+01,  7.24880e+00, 10, 5, 10, 5,
#        -8.02100e-01,  1.00690e+00,  3.70660e+00, -3.12400e-01,
#         5.89600e-01, -4.94000e-01,  6.01300e-01, -1.97200e-01,
#        -2.91000e-02,  1.92000e-02, -1.00000e-04, -6.62000e-02])
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
        files_to_copy.extend(glob.glob('%s/V*H0He3.*.txt' % data_dir))
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
n_knots['H-He p'] = 3
n_knots['He-W p'] = 3
n_knots['He-H p'] = 3
n_knots['He-He p'] = 3
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

rsamples_folder = os.path.join(save_folder, 'Full_Optimize') 

if not os.path.exists(rsamples_folder) and proc_id == 0:
    os.mkdir(rsamples_folder)
    
comm.Barrier()  

bounds = (
            (1, 7),  (0, 2),
            (0, 1), (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.5, 0.5),
            (0, 1), (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.5, 0.5),
            (0, 1), (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.5, 0.5),
            (0, 1), (-1, 1), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.5, 0.5),
            (-3, 0),  ( 1, 5), (-2, 5), (-1, 0.2), (-1, 2), (-3, 3),
            (0.2, 1), (-1, 0), (-0.1, 0.1), (0, 0.2), (-0.5, 0.5), (-0.5, 0.5),

            )       


data_ref = np.loadtxt('dft_yang.txt')


pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/init.eam.he')

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm_split, proc_id, param_dict['work_dir'])

loss = np.empty((0))
samples = np.empty((0, 14))


for i in range(112):
    _loss = np.loadtxt('Fitting_Output_FS_Final/Genetic_Algorithm/Loss_%d.txt' % i)
    _samples = np.loadtxt('Fitting_Output_FS_Final/Genetic_Algorithm/Samples_%d.txt' % i)

    loss = np.hstack([loss, _loss])
    samples = np.vstack([samples, _samples])

idx = np.argsort(loss)
loss = loss[idx]
samples = samples[idx]

if proc_id == 0:
    
    x0 =  np.array([
                1.7015e+00,  4.2490e-01, 
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                -1.1982e+00,  3.1443e+00, -3.2970e-01, -2.2820e-01,  4.1590e-01, -4.7750e-01 ,
                6.8130e-01, -3.8090e-01,  6.3500e-02,  8.6000e-03,  -9.4000e-03, 1.3100e-02
                ])
else:
    set = True
    while set:
        x_trial = samples[np.random.randint(low = 1, high = 1000)]

        if x_trial[0] > 1:
            x0 = np.hstack([ x_trial[:2],
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            x_trial[2:]
                            ])
            break

print('x_init')

optimize.differential_evolution(He_Fitting.loss_func, bounds, args=(data_ref, eam_fit, False, True, rsamples_folder),
                                init='latinhypercube', mutation=1.5, recombination=0.25, popsize=50, maxiter=50, polish=True, x0=x0)