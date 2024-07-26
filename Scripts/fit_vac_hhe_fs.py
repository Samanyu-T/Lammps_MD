import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
import matplotlib.pyplot as plt
import math
import FS_Fitting_Serial, Handle_PotFiles_FS
from scipy.optimize import minimize

def sim_h_he(r, potfile,type='fs'):
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
    if type=='alloy':
        lmp.command('pair_style eam/alloy' )
    elif type=='fs':
        lmp.command('pair_style eam/fs')
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    # lmp.command('write_dump all custom dump.%i.atom id type x y z' % i)
    pe = lmp.get_thermo('pe') / lmp.get_natoms()
    
    return pe

def loss_func(x, eam_fit, data_dft):

    Zh = 1
    Zhe = 2
    A = 5.46
    h = 0.9
    a0 = 0.529
    k = 0.1366
    # x = np.hstack([A, Zh, x[0], Zhe, x[1], x[2:]])
    loss = 1e6 *( (x[0] > 1) + (x[2] > 2) )

    x = np.hstack([A, x])

    eam_fit.sample_to_file(x)

    Handle_PotFiles_FS.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])


    for i, row in enumerate(data_dft):
        r, dft_pe = row
    
        pot_pe = sim_h_he(r, eam_fit.lammps_param['potfile'])

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
zbl = FS_Fitting_Serial.ZBL(2, 1)
y = zbl.eval_zbl(r)

# data_dft = np.hstack([r.reshape(-1,1), y.reshape(-1,1)])
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/beck.eam.fs')


n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 0
n_knots['He-H p'] = 2
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 4

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

eam_fit = FS_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])


# x = np.array([-1.875e-01,  2.568e-01, -2.578e-01, -1.852e-02,  2.622e-02, -3.972e-02])

x = np.array([ 1, 1.202e+00 , 2, 3.884e-01, -1.103e-01,  1.809e-01, -6.601e-01, -2.979e-02 , 2.880e-02, -6.059e-02])

# x_res = minimize(loss_func, x, args=(eam_fit, data_dft), method='Powell',options={"maxiter":1000}, tol=1e-4)
# print(x_res)
# x = x_res.x


Zh = 1
Zhe = 2
A = 5.46
h = 0.9
a0 = 0.529
k = 0.1366

x = np.hstack([A, x])

# x = np.hstack([A, Zh, x[0], Zhe, x[1], x[2:]])

# x = np.array([ 4.92700000e+00 , 2.48348537e+02,  7.53497164e+00,  2.79972926e-01,
#   5.59966304e-01, -6.27399559e+00,  2.27451317e-02, -5.81315963e-02,
#   2.31961952e-01])

x = np.array([ 5.46 ,  0.91861628,  1.27351374,  1.99999996,  0.78466961, -0.1294734, 
              0.1495497,  -0.36530064, -0.03049166,  0.03071201, -0.06344354])

x = np.array([5.46 ,  0.91861628,  1.27351374, 1.99999996,  0.78466961,-3.66286476e-01,  1.46948597e-01, -3.65438537e-01, -3.32882749e-03, 2.06810423e-02, -3.23230052e-01])

eam_fit.sample_to_file(x)

Handle_PotFiles_FS.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_FS.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.fs')




r_plt = np.linspace(0.5, 4, 100)
zbl = FS_Fitting_Serial.ZBL(2, 1)
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

zbl = FS_Fitting_Serial.ZBL(2, 1)
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



n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 0
n_knots['He-H p'] = 2
n_knots['He-He p'] = 2
n_knots['W-He'] = 0
n_knots['He-He'] = 4
n_knots['H-He'] = 4

pot, potlines, pot_params = Handle_PotFiles_FS.read_pot('git_folder/Potentials/beck.eam.fs')

eam_fit = FS_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

x = np.array([ 5.46 ,  0.91861628,  1.27351374,  2,  0.78466961, 2, 3e-4, 
              -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, -0.1294734, 
              0.1495497,  -0.36530064, -0.03049166,  0.03071201, -0.06344354])

eam_fit.sample_to_file(x)

Handle_PotFiles_FS.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_FS.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.fs')