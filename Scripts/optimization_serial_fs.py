import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import He_Fitting
import Handle_PotFiles_He
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

# 0.68089212  0.77818806 -8.22123366  8.35483589 -0.75619651  1.65186436
#   1.97897902 -0.32667374  0.435886    0.06747498

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
        files_to_copy.extend(glob.glob('%s/V*H0He4.*.txt' % data_dir))

    if h_he:
        # files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He0.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H1He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H1He3.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H2He0.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H2He1.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H2He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H2He3.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H3He0.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H3He1.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H3He2.*.txt' % data_dir))
        # files_to_copy.extend(glob.glob('%s/V*H3He3.*.txt' % data_dir))
    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)
# 0.757998409172  ,-0.149264550264  ,0.615848584083  ,-3.409149303432  ,2.913708564876  ,-0.098886153644  ,0.075722340000  ,-0.027001780000  ,
comm = MPI.COMM_WORLD

proc_id = 0

n_procs = 1
 
pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/final.eam.he')

# 1.510231228597197  ,-1.468340827434945  ,4.087238819276577  ,2.197917203259067  ,2.184035325825760  ,-0.416439283365776  ,0.396477830267865  ,-0.035264094255146  ,

# 0.48309483 -0.11809125  2.10966551 -0.33984664 28.67842118  0.41549807
#  -1.9733587   3.66449741  1.60497915 -0.68848128  0.80103327 -0.74969712
#  -0.19581129  0.37283166 -0.41408367 -0.03112508  0.05222353 -0.15153377
n_knots = {}
n_knots['He F'] = 0
n_knots['H-He p'] = 0
n_knots['He-W p'] = 3
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
# 1.509145543093065  ,-1.463700330113199  ,4.083683514861796  ,2.213287790918133  ,2.168181352311828  ,-0.418858034035005  ,0.399063366842790  ,-0.035033031442598  ,
# 1.527317670536826  ,-1.487419794162301  ,4.201509188838966  ,2.204655472293622  ,2.206021527173542  ,-0.409756736495524  ,0.391280786005535  ,-0.035724816849591  ,
# 2.51029847 -0.34896591  0.47865807 -0.3762      3.23425928 -0.0276
#   0.04344    -0.0747
n_knots['He-He'] = 0
n_knots['H-He'] = 0
n_knots['W-He p'] = 0
# -0.111008003931170  ,0.257370491090823  ,0.005965998599381  ,2.449325455281385  ,-0.006037000006585  ,0.072594422931774  ,-0.038927822536880  ,
with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

copy_files(True, True, True, param_dict['work_dir'], param_dict['data_dir'])

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

sample2 = eam_fit.gen_rand()

sample = np.loadtxt('sample.txt') 

'''Samples '''
# 4.04418578  1.5686086   1.75723945 -1.38616247  1.5895147   2.24968674
#   2.72359989 -0.11628429  0.17846521 -0.69014962  0.46294404 -0.08422526
#  -0.06079003  3.2733298   0.00868395  0.05599773  0.21078752

# 4.97167138e+00  8.71373715e-01  2.24725834e+00  3.50801232e-01 -7.52852715e-01  1.94688228e+00 -3.11534783e-03 -6.84004563e-02  2.25654264e-01  1.12128196e+00 -2.35231674e+00  2.68201258e+00  5.32830000e+00  2.87950000e+00 -1.84800000e-01  2.89500000e-01 -8.27800000e-01  2.84200000e-01 -5.88000000e-01 -1.11100000e-01  3.28350000e+00  2.72000000e-02 -7.88000000e-02 -2.88000000e-02

# 5.88150144e+00  8.79567188e-01  3.11368088e+00  3.61655251e-01 -5.04077537e-01  1.94688228e+00 -3.10890120e-03 -6.84004533e-02  2.25654264e-01  1.12125439e+00 -2.36179526e+00  2.69050899e+00  5.32829999e+00  2.87950000e+00 -1.84800000e-01  2.89500000e-01 -8.27800000e-01  2.84200000e-01 -5.88000000e-01 -1.11100000e-01  3.28350000e+00  2.72000000e-02 -7.88000000e-02 -2.88000000e-02
# 5.95210384  0.56392213  0.5754907 -0.57020189 -0.07348795  2.14479997 -0.03072961 -0.01334507  0.25942299  1.50669441 -1.27460585  3.74833262  2.01991162  2.20357552 -0.42493117  0.37492672 -0.02326983  0.45681563  -0.4344291  -0.0468169   3.27625827 -0.01210053 -0.08589632  0.10210008
# 5.54383005  0.57615736 -0.46529765 -0.56742808 -0.07346408  2.1448
#  -0.0314     -0.0138      0.26069786  1.5067     -1.2939      3.605
#  -1.7596      2.2029     -0.4349      0.3961     -0.0432      0.4542
#  -0.435      -0.0468      3.2763     -0.0121     -0.0859      0.1021

# sample[eam_fit.map['He-H p']] = np.array([[ 0.6309, -2.1765,  0.5955,  -0.0599, 0.4942, 0.2246]])
#  1.15577835  1.37018979  1.14701264  0.91343181  2.26586415  1.0037183
#  -1.89165792  2.74925153  1.88329318 -0.63323941  0.63944221 -0.32335614
#  -0.10747245  0.13973368 -0.36959604 -0.02580016  0.0362517  -0.05010032

# 5.81445987  8.64945015  1.12669639  4.42586009  0.26934147 -1.8554974
#   2.99477294  1.79655751 -0.75707402  0.74006193 -0.13316692
# sample = np.array([1e-4])
# sample += 1e-2*np.random.random(sample.shape)
print(sample2.shape, sample.shape)

data_ref = np.loadtxt('dft_yang.txt')

t1 = time.perf_counter()

eam_fit.sample_to_file(sample)

whe = eam_fit.pot_lammps['W-He']

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])

whe = whe[1:]/r[1:]

plt.plot(r[201:], whe[200:])

plt.show()
plt.plot(r, pot['W-He p'], label='W-He')
plt.plot(r, pot['H-He p'], label='H-He')
plt.plot(r, pot['He-W p'], label='He-W')
plt.plot(r, pot['He-H p'], label='He-H')
plt.plot(r, pot['He-He p'], label='He-He')
plt.legend()
plt.show()

# print(sample[eam_fit.map['He-H p']])
He_Fitting.simplex(n_knots, comm, proc_id, sample, 10000, param_dict['work_dir'], param_dict['save_dir'], True)

t2 = time.perf_counter()
print(t2 - t1)

