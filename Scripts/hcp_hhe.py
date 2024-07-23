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

def sim_hcp_helium(filepath, potfile):
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

def loss_func(x, eam_fit, data_dft):
    # A = x[0]
    # Z = x[1]
    Z = 2
    A = 5.4
    a0 = 0.529
    k = 0.1366
    x = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x])
    eam_fit.sample_to_file(x)

    Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

    loss = 100*(Z > 2) + 100*( A > 5.8) + 100*(A < 5.4)

    for i, row in enumerate(data_dft):
        lat, dft_stress, dft_pe = row

        pot_stress, pot_pe = sim_hcp_helium('HCP_H_He_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])

        # if dft_stress <20:
        #     loss += lat*(1 - pot_stress/dft_stress)**2
        # else:
        #     loss += 1e-1*(1 - pot_stress/dft_stress)**2

        loss +=  lat*(1 - pot_pe/dft_pe)**2

    print(loss)
    return loss

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

data_dft = np.hstack([stress_dft, hcp_dft])

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

# potfile =  'Fitting_Runtime/Potentials/optim.0.eam.alloy' 
eam_fit = EAM_Fitting_Serial.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

# x = np.array([-4.078e-01, 6.777e-01, -1.038e+00, -2.816e-02,  4.515e-02, -7.875e-02])

# Z = 1.993
# A = 4.9
# x = np.hstack([A, Z, x])

# x =  np.array(  [-3.670e-01,  4.788e-01, -3.763e-01, -2.759e-02, 4.339e-02, -7.454e-02])

x = np.array([-2.220e-01,  9.288e-01, -3.680e+00 ,-2.914e-02 , 2.506e-02, -4.401e-02])

x_res = minimize(loss_func, x, args=(eam_fit, data_dft), method='Powell',options={"maxiter":100}, tol=1e-4)
print(x_res)
x = x_res.x

# loss_func(x, eam_fit, data_dft)
stress_arr = np.zeros((len(data_dft,)))
pe_arr = np.zeros((len(data_dft,)))


A = 5.4
Z = 2
a0 = 0.529
k = 0.1366
sample = np.hstack([A, Z**4/(k*np.pi*a0**3), (2*Z/a0), x])

print(sample)

eam_fit.sample_to_file(sample)

Handle_PotFiles.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

for i, row in enumerate(data_dft):
    lat, dft_stress, dft_pe = row

    pot_stress, pot_pe = sim_hcp_helium('HCP_H_He_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])
    
    stress_arr[i]  = pot_stress
    pe_arr[i] = pot_pe
    # loss += (1 - pot_stress/dft_stress)**2
    # loss += (1 - pot_pe/dft_pe)**2# # potfile = 'He-Beck1968_modified.table'

# lat_arr = np.linspace(1.3, 4, 20)
# pe_arr = np.zeros(lat_arr.shape)
# stress_arr = np.zeros(lat_arr.shape)
# vol_arr = np.zeros(lat_arr.shape)



# conv = 0.602214

# vol_ref = conv*np.array([3.944, 3.871, 3.772, 3.422])

plt.loglog(data_dft[:,0], stress_arr, label='fit')
plt.loglog(data_dft[:,0], data_dft[:,1], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Hydrostatic Stress / GPA')
plt.title('Pressure-Volume curve of a HCP Hydrogen-Helium Lattice')
plt.legend()
plt.show()

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])

plt.plot(data_dft[:,0], pe_arr, label='fit')
plt.plot(data_dft[:,0], data_dft[:,2], label='dft', color='black')
plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Energy-Lattice Constant curve of a HCP Hydrogen-Helium Lattice')
plt.legend()
plt.show()


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


pot = eam_fit.pot_lammps['H-He'][1:]
pot /= r[1:]
plt.ylabel('Pairwise Potential / eV')
plt.xlabel('Distance/ A')
plt.title('H-He Pairwise Potential')
plt.plot(r[201:], pot[200:])
print(pot.min())
plt.show()

