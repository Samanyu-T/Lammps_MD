import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import FS_Fitting
import Handle_PotFiles_FS
import time
import json, glob, shutil
from mpi4py import MPI
import matplotlib.pyplot as plt

# Poor H-He
# [ 1.02469398e+01  1.27351374e+00  3.39815554e+00  1.11246492e+00
#   6.76706812e+00  7.49846373e-01  3.71834135e+00  3.36557055e-04
#  -1.78010433e+00  3.15816754e+00  1.85458969e+00 -6.57323025e-01
#   6.04536904e-01 -2.84569871e-01 -3.64470312e-01  4.89139790e-01
#  -3.60373997e-01 -2.75840515e-02  4.34467205e-02 -7.48828497e-02
#  -2.58458425e-01  1.10183324e+00 -3.77181683e-01  7.69354047e-02
#   5.50140079e-02 -3.14586329e-01]


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

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/init.eam.fs')

# 5.60697527e+00  3.85118169e+00  9.49832030e+01  3.84392370e+00  1.19397215e+00  2.30219362e+01  7.76016391e-01  2.16019733e+00  1.45467904e+00 -1.85564438e+00  3.01824645e+00  1.86007434e+00 -6.61953938e-01  6.11439256e-01 -3.11273002e-01 -4.14029651e-01  6.77237863e-02 -3.78793307e-01  8.04632485e-01  1.49701602e+00 -1.10496938e-01 -1.01947712e-01  1.84336665e-01 -3.20069363e-01 -4.21210361e-02  3.50947646e-02  4.49373636e-02

# 0.48309483 -0.11809125  2.10966551 -0.33984664 28.67842118  0.41549807
#  -1.9733587   3.66449741  1.60497915 -0.68848128  0.80103327 -0.74969712
#  -0.19581129  0.37283166 -0.41408367 -0.03112508  0.05222353 -0.15153377
n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 0
n_knots['He-W p'] = 2
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 4
n_knots['He-He'] = 0
n_knots['H-He'] = 0
n_knots['W-He p'] = 3

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = FS_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

x_init = np.loadtxt('x_init_new.txt')

loss = []

data_ref = np.loadtxt('dft_yang.txt')

for _x in x_init:
    _loss = FS_Fitting.loss_func(_x, data_ref, eam_fit, False, False, None)
    print(_loss, _x)
    loss.append(_loss)

loss = np.array(loss)

np.savetxt('loss_new.txt', loss)