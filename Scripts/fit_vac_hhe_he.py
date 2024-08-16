import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
import matplotlib.pyplot as plt
import math
import He_Fitting, Handle_PotFiles_He
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def sim_h_he(r, potfile,type='he'):
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
    lmp.command('pair_style eam/%s' % type )
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    lmp.command('write_dump all custom dump.%i.atom id type x y z' % i)
    pe = lmp.get_thermo('pe')
    
    return pe

def analytical_h_he(x, eam_fit, data_dft):

    x = np.hstack([5.5, 0.6, x])
    eam_fit.sample_to_file(x)

    r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])
    rho = np.linspace(eam_fit.pot_params['rhomin'], eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho'])

    rho_h_he = interp1d(r, eam_fit.pot_lammps['H-He p'])

    rho_he_h = interp1d(r,eam_fit.pot_lammps['He-H p'])

    F_h = interp1d(rho,eam_fit.pot_lammps['H F'])

    F_he = interp1d(rho,eam_fit.pot_lammps['He F'])

    zbl_hhe = He_Fitting.ZBL(2, 1)

    r_plt = data_dft[:,0]

    coef_dict = eam_fit.fit_sample(x)

    pot_hhe = zbl_hhe.eval_zbl(r_plt) + He_Fitting.splineval(r_plt, coef_dict['H-He'], eam_fit.knot_pts['H-He'])

    emd_H_He = np.zeros(r_plt.shape)
    emd_He_H = np.zeros(r_plt.shape)

    for i, _r in enumerate(r_plt):

        _rho_h_he = rho_h_he(_r)

        emd_H_He[i] = F_he(_rho_h_he)
        
        _rho_h_he = rho_he_h(_r)

        emd_He_H[i] = F_h(_rho_h_he)

    total_hhe = (emd_H_He + emd_He_H + pot_hhe)

    loss = np.linalg.norm(total_hhe - data_dft[:,1])
    
    print(x, loss)

    return loss

def loss_func(x, eam_fit, data_dft):

    Zh = 1
    Zhe = 2
    A = 5.5
    h = 0.9
    a0 = 0.529
    k = 0.1366
    # x = np.hstack([A, Zh, x[0], Zhe, x[1], x[2:]])
    loss = 1e6 *( (x[0] > 1) + (x[2] > 2) )

    x = np.hstack([A, 0.6, x])

    eam_fit.sample_to_file(x)

    Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])


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
zbl = He_Fitting.ZBL(2, 1)
y = zbl.eval_zbl(r)

# data_dft = np.hstack([r.reshape(-1,1), y.reshape(-1,1)])
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/beck.eam.he')


n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 0
n_knots['He-H p'] = 2
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 4
n_knots['W-He p'] = 0

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])


# x = np.array([-1.875e-01,  2.568e-01, -2.578e-01, -1.852e-02,  2.622e-02, -3.972e-02])

x = np.array([ 1,  4, 1e-3 , 4.7, 4, 1,
              -1.26327399e-01,  1.31954504e-01, -2.34140120e-01, -2.33344481e-02,  2.75030830e-02, -4.83606197e-02])

print(eam_fit.gen_rand().shape, x.shape)
eam_fit.sample_to_file(np.hstack([5.5, 0.6, x]))
r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]
hhe = eam_fit.pot_lammps['H-He'][1:]
hhe = hhe / r
plt.plot(r[400:], hhe[400:])
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')
plt.show()

x = np.array([1.36606702e+00,  4.35316203e+00, 1.00000000e-03,
             -2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
             -1.12250698e-01,  3.95622407e-02,  1.49297332e-01, -2.38165659e-02, 2.79419759e-02, -5.00556693e-02])

x_res = minimize(analytical_h_he, x, args=(eam_fit, data_dft), method='Powell',options={"maxfev":10}, tol=1e-4)
print(x_res)
x = x_res.x

x = np.hstack([5.5, 0.6, x_res.x])

eam_fit.sample_to_file(x)

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.he')




# r_plt = np.linspace(0.5, 4, 100)
# zbl = He_Fitting.ZBL(2, 1)
# y = zbl.eval_zbl(r)

r_plt = data_dft[:, 0]

pe_arr = np.zeros((len(r_plt,)))

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

zbl = He_Fitting.ZBL(2, 1)
plt.plot(r_plt, pe_arr, label='full inc eam')
plt.plot(data_dft[:,0], zbl.eval_zbl(data_dft[:,0]), label = 'zbl')  
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')

pot = eam_fit.pot_lammps['H-He'][1:]/r
plt.plot(r[125:], pot[125:], label='pairwise component')

plt.scatter(h_he_ref[:,0], h_he_ref[:,1], label='qmc', color='red')

plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Vacuum Interaction of H-He')
plt.legend()
plt.show()

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])
rho = np.linspace(eam_fit.pot_params['rhomin'], eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho'])

rho_h_he = interp1d(r, eam_fit.pot_lammps['H-He p'])

rho_he_h = interp1d(r,eam_fit.pot_lammps['He-H p'])

F_h = interp1d(rho,eam_fit.pot_lammps['H F'])

F_he = interp1d(rho,eam_fit.pot_lammps['He F'])

zbl_hhe = He_Fitting.ZBL(2, 1)

r_plt = data_dft[:,0]

coef_dict = eam_fit.fit_sample(x)

pot_hhe = zbl_hhe.eval_zbl(r_plt) + He_Fitting.splineval(r_plt, coef_dict['H-He'], eam_fit.knot_pts['H-He'])

emd_H_He = np.zeros(r_plt.shape)
emd_He_H = np.zeros(r_plt.shape)

for i, _r in enumerate(r_plt):

    _rho_h_he = rho_h_he(_r)

    emd_H_He[i] = F_he(_rho_h_he)
    
    _rho_h_he = rho_he_h(_r)

    emd_He_H[i] = F_h(_rho_h_he)

total_hhe = (emd_H_He + emd_He_H + pot_hhe)
np.savetxt('h_he_pairwise.txt',data_dft)
plt.plot(r_plt, total_hhe, label='total')
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')
plt.plot(r_plt, emd_H_He, label='Electron density of Hydrogen on Helium')
plt.plot(r_plt, emd_He_H, label='Electron density of Helium on Hydrogen')
plt.plot(r_plt, pot_hhe, label='Covalent')

plt.ylabel('Contributions / eV')
plt.xlabel('Distance/ A')
plt.title('H-He Pairwise Potential')

plt.legend()
plt.show()

plt.plot(eam_fit.pot_lammps['H-He p'])
plt.plot(eam_fit.pot_lammps['He-H p'])
plt.show()

plt.plot(rho, eam_fit.pot_lammps['He F'])
plt.show()



n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 2
n_knots['He-W p'] = 2
n_knots['He-H p'] = 2
n_knots['He-He p'] = 2
n_knots['W-He'] = 4
n_knots['He-He'] = 4
n_knots['H-He'] = 4
n_knots['W-He p'] = 3

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/beck.eam.he')

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])


# x = np.array([ 2.25309532e+00,  5.78508321e-01,
#                1.36606702e+00,  4.35316203e+00, 1.00000000e-03,
#                4.78434000e+01,  6.79830000e+00, 1e-3,
#                -2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
#                 0, 0, 0,
#               -8.78443579e-01,  1.48624430e+00,  2.27126353e+00, -3.40994311e-01,  5.90983069e-01, -2.18796556e-01,
#               -3.67006528e-01,  4.78912258e-01, -3.76192674e-01, -2.75952106e-02,  4.34246773e-02, -7.47000749e-02,
#               -1.12250698e-01,  3.95622407e-02,  1.49297332e-01, -2.38165659e-02, 2.79419759e-02, -5.00556693e-02])

x = np.array([   1.7015e+00,  4.2490e-01, 
                 1.36606702e+00,  4.35316203e+00, 1.00000000e-03,
                 0, 0, 0,
                 2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
                 0, 0, 0,
                -1.1982e+00,  3.1443e+00, -3.2970e-01, -2.2820e-01,  4.1590e-01, -4.7750e-01 ,
                -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
                -0.12564656, 0.13166891, -0.23713911, -0.02355287 , 0.02697471 ,-0.04887022,
                 6.8130e-01, -3.8090e-01,  6.3500e-02,  8.6000e-03,  -9.4000e-03, 1.3100e-02])

# x = np.array([   6.08600e-01,  2.78500e-01, 
#                  2.53408e+01,  6.75140e+00, 0,
#                  0, 0, 0,
#                  2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
#                  0, 0, 0,
#                 -6.46700e-01,   1.13620e+00 ,-1.34460e+00, -3.12900e-01,  5.53600e-01,  4.46000e-02 ,  6.57300e-01,
#                 -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
#                 -0.12564656, 0.13166891, -0.23713911, -0.02355287 , 0.02697471 ,-0.04887022,
#                  6.57300e-01, -4.56100e-01, -5.06000e-02,  2.86000e-02, -1.43000e-02,  -1.01000e-02])
eam_fit.sample_to_file(x)

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.he')

plt.plot(rho[:1000], eam_fit.pot_lammps['H F'][:1000])
plt.plot(rho[:1000], eam_fit.pot_lammps['He F'][:1000])
plt.plot(rho[:1000], eam_fit.pot_lammps['W F'][:1000])

plt.show()