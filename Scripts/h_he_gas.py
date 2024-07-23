import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
import matplotlib.pyplot as plt
import math
import EAM_Fitting_Serial, Handle_PotFiles
from scipy.optimize import minimize

def sim_h_he(r, potfile):
    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
    lmp.command('lattice fcc 20')
    lmp.command('region r_simbox block 0 1 0 1 0 1 units lattice')
    lmp.command('region r_atombox block 0 1 0 1 0 1 units lattice')
    lmp.command('create_box 3 r_simbox')
    lmp.command('create_atoms 3 single 0 0 0')
    lmp.command('create_atoms 2 single 0 0 %f units box' % r)
    lmp.command('mass 1 183.84')
    lmp.command('mass 2 1.00784')
    lmp.command('mass 3 4.002602')
    lmp.command('pair_style eam/alloy' )
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    # lmp.command('write_dump all custom dump.%i.atom id type x y z' % i)
    pe = lmp.get_thermo('pe') / lmp.get_natoms()
    
    return pe

def loss_func(x, eam_fit, data_dft):
    # A = x[0]
    # Z = x[1]
    Z = 2
    A = 5.4
    a0 = 0.529
    k = 0.1366
    loss = 0

    x = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x])
    eam_fit.sample_to_file(x)

    Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])


    for i, row in enumerate(data_dft):
        r, dft_pe = row
    
        pot_pe = sim_h_he(r, eam_fit.lammps_param['potfile'])

        if r > 1.4:
            loss += (1 - pot_pe/dft_pe)**2
    print(x, loss)
    return loss

data_dft = []

with open('hhe_energy.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        r =  float(split[0][-3:])
        pe = float(split[-1])
        data_dft.append([r, pe])
data_dft = np.array(data_dft)

r = np.linspace(1.5, 4, 100)
zbl = EAM_Fitting_Serial.ZBL(2, 1)
y = zbl.eval_zbl(r)

# data_dft = np.hstack([r.reshape(-1,1), y.reshape(-1,1)])
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles.read_pot('git_folder/Potentials/beck.eam.alloy')


n_knots = {}
n_knots['He_F'] = 2
n_knots['He_p'] = 2
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 4


with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

eam_fit = EAM_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

x = np.array([-1.875e-01,  2.568e-01, -2.578e-01, -1.852e-02,  2.622e-02, -3.972e-02])

x = np.array([ -1.57304977e-01, 2.55730610e-01, -3.04095309e-01, -2.91073705e-02,  3.33967828e-02, -1.06016653e-01])
x = np.array([ -2.43262355e-01, 3.72114002e-01,  7.85573717e-02, -2.85585122e-02,  3.96325522e-02, -1.65936981e-01]) 
x = np.array([-4.91023057e-02, 6.44896465e-02, -4.29839368e-01, -2.90499863e-02,  2.89994789e-02, -7.50996183e-02])
x = np.array([-2.96808939e-01, 1.12443859e+00, -3.83569086e+00, -2.86488137e-02,  3.27661217e-02,-1.09961765e-0])
x = np.array([-2.220e-01,  9.288e-01, -3.680e+00 ,-2.914e-02 , 2.506e-02, -4.401e-02])
# x = np.array([-3.682e+00,  8.656e+00, -1.773e+01, -1.607e-01,  4.941e-01, -1.375e+00, -1.930e-02,  2.345e-02, -1.243e-01])
# x = np.array([-3.67194357e+00, 9.69802650e+00, -3.58810962e+00, -5.76257268e-02,  3.45931845e-02,
#   1.59252021e-01 ,-1.88672790e-02,  2.32253675e-02 ,-1.48271492e-01])

# x = np.array([2.79972926e-01, 5.59966304e-01, -6.27399559e+00,  2.27451317e-02, -5.81315963e-02, 2.31961952e-01])

# x_res = minimize(loss_func, x, args=(eam_fit, data_dft), method='Powell',options={"maxiter":1000}, tol=1e-4)
# print(x_res)
# x = x_res.x

Z = 2
A = 5.4
a0 = 0.529
k = 0.1366

x = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x])

# x = np.array([ 4.92700000e+00 , 2.48348537e+02,  7.53497164e+00,  2.79972926e-01,
#   5.59966304e-01, -6.27399559e+00,  2.27451317e-02, -5.81315963e-02,
#   2.31961952e-01])

eam_fit.sample_to_file(x)

Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.alloy')




r_plt = np.linspace(0.5, 4, 100)
zbl = EAM_Fitting_Serial.ZBL(2, 1)
y = zbl.eval_zbl(r)

pe_arr = np.zeros((len(r,)))

for i, _r in enumerate(r_plt):
    pot_pe = sim_h_he(_r, eam_fit.lammps_param['potfile'])
    
    pe_arr[i] = pot_pe

h_he_ref = np.array([
            [ 2.64885000e+00,  5.92872325e-03],
            [ 2.91373500e+00,  1.38739018e-03],
            [ 3.17862000e+00, -3.86056397e-04],
            [ 3.44350500e+00, -5.48062207e-04],
            [ 3.70839000e+00, -5.85978460e-04],
            [ 3.97327500e+00, -4.22249185e-04],
            [ 4.23816000e+00, -3.75715601e-04],
            [ 4.76793000e+00, -1.68037941e-04],
            ])

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]

zbl = EAM_Fitting_Serial.ZBL(2, 1)
plt.plot(r_plt, pe_arr, label='full inc eam')
plt.plot(data_dft[:,0], zbl.eval_zbl(data_dft[:,0]), label = 'zbl')  
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')

pot = eam_fit.pot_lammps['H-He'][1:]/r
plt.plot(r[125:], pot[125:], label='pairwise component')

plt.scatter(h_he_ref[:,0], h_he_ref[:,1], label='qmc', color='red')

plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Pairwise Interaction of H-He')
plt.legend()
plt.show()

plt.plot(r[80:], pot[80:], label='pairwise')
plt.plot(r[80:], zbl.eval_zbl(r[80:]), label='zbl')
plt.show()