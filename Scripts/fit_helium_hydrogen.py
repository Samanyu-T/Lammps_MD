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
from scipy.interpolate import interp1d

def sim_hcp(filepath, potfile):
    lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('read_data %s' % filepath)
    lmp.command('pair_style eam/alloy' )
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    lmp.command('minimize 1.0e-4 1.0e-6 100 1000')
    # lmp.command('write_dump all custom dump.atom id type x y z' )
    hydrostatic_pressure = 1e-4*(lmp.get_thermo('pxx') +  lmp.get_thermo('pyy') +  lmp.get_thermo('pzz') )/3
    pe = lmp.get_thermo('pe') / lmp.get_natoms()
    
    return hydrostatic_pressure, pe

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

def loss_func(x, eam_fit, dft_hcp_helium, dft_hcp_hhe, dft_vacuum_hhe):

    A = 5.4
    Z = 2
    a0 = 0.529
    k = 0.1366
    x_he =  np.array( [-3.670e-01,  4.788e-01, -3.763e-01, -2.759e-02, 4.339e-02, -7.454e-02] )
    
    x_he = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x_he])
    
    x = np.hstack([x_he, x])

    eam_fit.sample_to_file(x)

    Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

    loss = 0

    # for i, row in enumerate(dft_hcp_helium):
    #     lat, dft_stress, dft_pe = row

    #     pot_stress, pot_pe = sim_hcp('HCP_Helium_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])

    #     if dft_stress <20:
    #         loss += lat*(1 - pot_stress/dft_stress)**2
    #     else:
    #         loss += 1e-1*(1 - pot_stress/dft_stress)**2

    #     loss += (1 - pot_pe/dft_pe)**2

    # for i, row in enumerate(dft_hcp_hhe):
    #     lat, dft_stress, dft_pe = row

    #     pot_stress, pot_pe = sim_hcp('HCP_H_He_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])

    #     if dft_stress <20:
    #         loss += lat*(1 - pot_stress/dft_stress)**2
    #     else:
    #         loss += 1e-1*(1 - pot_stress/dft_stress)**2

    #     loss += (1 - pot_pe/dft_pe)**2

    for i, row in enumerate(dft_vacuum_hhe):
        r, dft_pe = row
    
        pot_pe = sim_h_he(r, eam_fit.lammps_param['potfile'])

        loss += 10*(1 - pot_pe/dft_pe)**2

    print(x[-9:], loss)
    return loss



dft_vacuum_hhe = []

with open('hhe_energy.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        r =  float(split[0][-3:])
        pe = float(split[-1])
        dft_vacuum_hhe.append([r, pe])
dft_vacuum_hhe = np.array(dft_vacuum_hhe)


stress_dft = []
with open('hcp_hhe_stress_curve.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        alat = float(split[0][-3:])
        pxx = float(split[2])
        pyy = float(split[2])
        pzz = float(split[2])

        stress = (pxx + pyy + pzz) / (3 * math.sqrt(2) * alat **3)

        conv = 1/1.60218e-2
        stress_dft.append([alat, stress*conv])

stress_dft = np.array(stress_dft)

hcp_dft = []

with open('hcp_hhe_total_energy.dat', 'r') as file:
    for line in file:
        split = line.split(' ')
        alat = float(split[0][-3:])
        pe = float(split[-1][:-2])/2 - 0.00138
        hcp_dft.append(pe)
hcp_dft = np.array(hcp_dft).reshape(-1, 1)

dft_hcp_hhe = np.hstack([stress_dft, hcp_dft])



stress_dft = []
with open('hcp_stress_curve.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        alat = float(split[0][-3:])
        pxx = float(split[2])
        pyy = float(split[2])
        pzz = float(split[2])

        stress = (pxx + pyy + pzz) / (3 * math.sqrt(2) * alat **3)

        conv = 1/1.60218e-2
        stress_dft.append([alat, stress*conv])

stress_dft = np.array(stress_dft)

hcp_dft = []

with open('hcp_total_energy.dat', 'r') as file:
    for line in file:
        split = line.split(' ')
        alat = float(split[0][-3:])
        pe = float(split[-1][:-2])/2 - 0.00138
        hcp_dft.append(pe)
hcp_dft = np.array(hcp_dft).reshape(-1, 1)

dft_hcp_helium = np.hstack([stress_dft, hcp_dft])


comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles.read_pot('git_folder/Potentials/beck.eam.alloy')


n_knots = {}
n_knots['He_F'] = 2
n_knots['He_p'] = 2
n_knots['W-He'] = 0
n_knots['He-He'] = 4
n_knots['H-He'] = 4


with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

# potfile =  'Fitting_Runtime/Potentials/optim.0.eam.alloy' 
eam_fit = EAM_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

x =  np.zeros((6, ))

x = np.array([-1.667e-01,  8.366e-01, -3.819e+00, -2.940e-02,  2.126e-02, -9.489e-03])

# x = np.array([-4.045e+00,  1.515e+01, -3.703e+01, -5.456e-02,  4.143e-02,
#            -1.908e-01, -1.996e-02,  1.611e-02,  3.742e-03])

# x = np.array([-4.21665994e+00,  1.06118691e+01, -2.50071398e+01, -1.70973301e-01, 5.39290007e-01, -1.50444481e+00, -1.92526923e-02,  2.40451210e-02, -1.34373665e-01] )
# x = np.array([-4.21665994e+00,  1.06118691e+01, -2.50071398e+01, -1.92526923e-02,  2.40451210e-02, -1.34373665e-01] )

# x = np.zeros((9, ))

# res = minimize(loss_func, x, args=(eam_fit, dft_hcp_helium, dft_hcp_hhe, dft_vacuum_hhe), method='Powell',options={"maxiter":1000}, tol=1e-4)
# print(res)
# x = res.x

A = 5.4
Z = 2
a0 = 0.529
k = 0.1366
x_he =  np.array( [-3.670e-01,  4.788e-01, -3.763e-01, -2.759e-02, 4.339e-02, -7.454e-02] )

x_he = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x_he])

sample = np.hstack([x_he, x])

print(sample)

eam_fit.sample_to_file(sample)

Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])
Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.alloy')

pot_hcp_helium = np.zeros(dft_hcp_helium.shape)

for i, row in enumerate(dft_hcp_helium):
    lat, dft_stress, dft_pe = row

    pot_stress, pot_pe = sim_hcp('HCP_Helium_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])
    pot_hcp_helium[i, 0] = lat
    pot_hcp_helium[i, 1]  = pot_stress
    pot_hcp_helium[i, 2] = pot_pe


pot_hcp_hhe = np.zeros(dft_hcp_hhe.shape)

for i, row in enumerate(dft_hcp_helium):
    lat, dft_stress, dft_pe = row

    pot_stress, pot_pe = sim_hcp('HCP_H_He_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])
    pot_hcp_hhe[i, 0] = lat
    pot_hcp_hhe[i, 1]  = pot_stress
    pot_hcp_hhe[i, 2] = pot_pe

pot_vacuum_hhe = np.zeros(dft_vacuum_hhe.shape)

for i, row in enumerate(dft_vacuum_hhe):
    r, dft_pe = row
    pot_vacuum_hhe[i, 0] = r
    pot_vacuum_hhe[i, 1] = sim_h_he(r, eam_fit.lammps_param['potfile'])


plt.plot(pot_hcp_helium[:,0], pot_hcp_helium[:,1], label='fit')
plt.plot(dft_hcp_helium[:,0], dft_hcp_helium[:,1], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Hydrostatic Stress / GPA')
plt.title('Pressure-Volume curve of a HCP Helium Lattice')
plt.legend()
plt.show()


plt.plot(pot_hcp_helium[:,0], pot_hcp_helium[:,2], label='fit')
plt.plot(dft_hcp_helium[:,0], dft_hcp_helium[:,2], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Energy-Lattice Constant curve of a HCP Helium Lattice')
plt.legend()
plt.show()


plt.plot(pot_hcp_hhe[:,0], pot_hcp_hhe[:,1], label='fit')
plt.plot(dft_hcp_hhe[:,0], dft_hcp_hhe[:,1], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Hydrostatic Stress / GPA')
plt.title('Pressure-Volume curve of a HCP Hydrogen-Helium Lattice')
plt.legend()
plt.show()


plt.plot(pot_hcp_hhe[:,0], pot_hcp_hhe[:,2], label='fit')
plt.plot(dft_hcp_hhe[:,0], dft_hcp_hhe[:,2], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Energy-Lattice Constant curve of a HCP Hydrogen-Helium Lattice')
plt.legend()
plt.show()


zbl = EAM_Fitting_Serial.ZBL(2, 1)
plt.plot(pot_vacuum_hhe[:,0], pot_vacuum_hhe[:,1], label='fit')
plt.plot(dft_vacuum_hhe[:,0], dft_vacuum_hhe[:,1], label='dft', color='black')
plt.plot(pot_vacuum_hhe[:,0], zbl.eval_zbl(pot_vacuum_hhe[:,0]), label = 'zbl')  

plt.xlabel('Radius / A')
plt.ylabel('Energy/ eV')
plt.title('Interaction of H-He within a vacuum')
plt.legend()
plt.show()

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])

pot = eam_fit.pot_lammps['He_F']
plt.ylabel('Embedding Function / eV')
plt.xlabel('Electron Density/ pot_units')
plt.title('Embedding function of Helium')
plt.plot(np.linspace(0, eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho']), pot)
plt.show()


pot = eam_fit.pot_lammps['He_p']
plt.ylabel('Electron Density / pot_units')
plt.xlabel('Distance/ A')
plt.title('Electron Density Function of Helium')
plt.semilogy(r, pot)
plt.show()


pot = eam_fit.pot_lammps['He-He'][1:]
pot /= r[1:]
plt.ylabel('Pairwise Potential / eV')
plt.xlabel('Distance/ A')
plt.title('He-He Pairwise Potential')
plt.plot(r[201:], pot[200:])
print(pot.min())
plt.show()


pot = eam_fit.pot_lammps['H-He'][1:]
pot /= r[1:]
plt.ylabel('Pairwise Potential / eV')
plt.xlabel('Distance/ A')
plt.title('H-He Pairwise Potential')
plt.plot(r[201:], pot[200:])
print(pot.min())
plt.show()

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])
rho = np.linspace(0, eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho'])

rho_h = interp1d(r, eam_fit.pot_lammps['H_p'])

rho_he = interp1d(r,eam_fit.pot_lammps['He_p'])

F_h = interp1d(rho,eam_fit.pot_lammps['H_F'])

F_he = interp1d(rho,eam_fit.pot_lammps['He_F'])

r_plt = np.linspace(0.9, 4, 100)

emd_H_He = np.zeros(r_plt.shape)
emd_He_H = np.zeros(r_plt.shape)

for i, _r in enumerate(r_plt):

    _rho = rho_h(_r)

    emd_H_He[i] = F_he(_rho)
    
    _rho = rho_he(_r)

    emd_He_H[i] = F_h(_rho)

plt.plot(pot_vacuum_hhe[:,0], pot_vacuum_hhe[:,1], label='total')
plt.plot(r_plt, emd_H_He, label='Electron density of Hydrogen on Helium')
plt.plot(r_plt, emd_He_H, label='Electron density of Helium on Hydrogen')
plt.plot(r[201:], pot[200:], label='Covalent')

plt.ylabel('Contributions / eV')
plt.xlabel('Distance/ A')
plt.title('H-He Pairwise Potential')

plt.legend()
plt.show()