import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import He_Fitting
import time
import json, glob, shutil
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

with open('fitting_genetic.json', 'r') as file:
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

rsamples_folder = os.path.join(save_folder, 'Genetic_Algorithm') 

if not os.path.exists(rsamples_folder) and proc_id == 0:
    os.mkdir(rsamples_folder)
    
comm.Barrier()  

t1 = time.perf_counter()
He_Fitting.genetic_alg(n_knots, comm_split, proc_id, work_dir, rsamples_folder)