import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import He_Fitting
import Handle_PotFiles_He
import time
import json, glob, shutil
from scipy import optimize
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

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/init.eam.he')

n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 0
n_knots['He-W p'] = 2
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 0

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

max_time = param_dict['max_time']
work_dir = param_dict['work_dir']
save_folder = param_dict['save_dir']
data_dir = param_dict['data_dir']

if proc_id == 0:
    copy_files(True, True, True, work_dir, data_dir)

comm.Barrier()

data_files_folder = os.path.join(work_dir, 'Data_Files')

lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

if os.path.exists(lammps_folder):

    shutil.rmtree(lammps_folder)

shutil.copytree(data_files_folder, lammps_folder)

# Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/init.eam.he')

# Call the main fitting class
eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

data_ref = np.loadtxt('dft_yang.txt')

color = proc_id % 8

optim_bounds = ( (color, color + 1), (0, 1), 
                 (1, 100), (0, 1), (1, 8),
                 (-3, -1), (-10, 10), (-20, 20), (-0.3, -0.1), (-1, 1), (-2, 2))

save_folder = os.path.join(param_dict['save_dir'], 'Python_Global_Optim')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)
comm.Barrier()

optimize.differential_evolution(He_Fitting.loss_func, bounds = optim_bounds, args=(data_ref, eam_fit, True, True, save_folder),
                                popsize=50)