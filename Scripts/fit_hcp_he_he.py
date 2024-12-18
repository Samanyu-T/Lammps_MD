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

def sim_hcp_helium(filepath, potfile, type='he'):
    lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('read_data %s' % filepath)
    lmp.command('pair_style eam/%s' % type)
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

    # A = 1.7015e+00
    # b = 4.2490e-01
    # x = np.hstack([A, b,  x])
    
    eam_fit.sample_to_file(x)

    Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

    loss = 0 #1e8*( (Z > 1.83) + ( A > 5.6) + (A < 5.4) + (h<0) + (h>0.78) ) 

    for i, row in enumerate(data_dft):
        lat, dft_stress, dft_pe = row

        pot_stress, pot_pe = sim_hcp_helium('HCP_Helium_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])

        if dft_stress <20:
            loss += lat*(1 - pot_stress/dft_stress)**2
        else:
            loss += 1e-1*(1 - pot_stress/dft_stress)**2

        loss +=  lat*(1 - pot_pe/dft_pe)**2

    print(x, loss)
    return loss

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

data_dft = np.hstack([stress_dft, hcp_dft])

comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/init.eam.he')


n_knots = {}
n_knots['He F'] = 0
n_knots['H-He p'] = 0
n_knots['He-W p'] = 0
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 0
n_knots['W-He p'] = 0

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

# potfile =  'Fitting_Runtime/Potentials/optim.0.eam.alloy' 
eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])

# x = np.array([-4.078e-01, 6.777e-01, -1.038e+00, -2.816e-02,  4.515e-02, -7.875e-02])

''' CURRENT STABLE OPTIMA '''
x =  np.array(  [1.62963646, -0.3658247,   0.48055115, -0.36578079,  3.23371698, -0.02758561, 0.04354402, -0.07569902])

x = np.array([1.644081838560815  ,-0.364089653944737  ,0.486755140426219  ,-0.370503077147278  ,3.238489844285963  ,-0.027611696678638  ,0.044106180101373  ,-0.075079529972039  ,])
x = np.array([0, 0, 0  ,1.800222855256981  ,0.000000000000000  ,0.000000000000000  ,0.000000000000000  ,]) 

# x = np.array([0.383091837975847  ,0.009079475509070  ,0.001077780079000  ,1.999949607675000  ,0.000000000000000  ,0.000000000000000  ,0.000000000000000  ,])
# x = np.array([1.644640508158  ,-0.362544854109  ,0.502218297628  ,-0.401297022906  ,3.232716794870  ,-0.027598698231  ,0.043451534835  ,-0.074820445456 ])
# x = np.hstack([1e-1*np.random.randn(3), 2, 1e-3*np.random.randn(3)])

# x_res = minimize(loss_func, x, args=(eam_fit, data_dft), method='Nelder-Mead',options={"maxiter":1000}, tol=1e-4)
# print(x_res)
# x = x_res.x

loss_func(x, eam_fit, data_dft)
stress_arr = np.zeros((len(data_dft,)))
pe_arr = np.zeros((len(data_dft,)))


# A = 1.7015e+00
# b = 4.2490e-01
# x = np.hstack([A, b,  x])

sample = x
print(sample)

eam_fit.sample_to_file(sample)

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

for i, row in enumerate(data_dft):
    lat, dft_stress, dft_pe = row

    pot_stress, pot_pe = sim_hcp_helium('HCP_Helium_DataFiles/lat.%.1f.data' % lat, eam_fit.lammps_param['potfile'])
    
    stress_arr[i]  = pot_stress
    pe_arr[i] = pot_pe



beck_stress_arr = np.zeros((len(data_dft,)))
beck_pe_arr = np.zeros((len(data_dft,)))

for i, row in enumerate(data_dft):
    lat, dft_stress, dft_pe = row

    pot_stress, pot_pe = sim_hcp_helium('HCP_Helium_DataFiles/lat.%.1f.data' % lat, potfile =  'git_folder/Potentials/WHHe_test.eam.alloy' , type='alloy')
    
    beck_stress_arr[i]  = pot_stress
    beck_pe_arr[i] = pot_pe


# conv = 0.602214

# press_ref = np.array([15.6, 16.2, 17.4, 23.3])
# lat_ref = np.array([2.1, 2.087, 2.069, 2.003])
# vol_ref = conv*np.array([3.944, 3.871, 3.772, 3.422])

# plt.loglog(lat_ref, press_ref, label='Experiment', color='green')
plt.semilogy(data_dft[:,0], stress_arr, label='our work')
plt.semilogy(data_dft[:,0], beck_stress_arr, label='Beck1968')
plt.semilogy(data_dft[:,0], data_dft[:,1], label='DFT', color='black')
plt.xlabel('Lattice Constant/ A', fontsize=14)
plt.ylabel('Hydrostatic Stress / GPA', fontsize=14)
plt.title('Pressure-Volume curve of a HCP Helium Lattice', fontsize=14)
plt.legend( fontsize=14)
plt.show()
np.savetxt('hcp_dft.txt',data_dft)
np.savetxt('stress_our.txt', stress_arr)
np.savetxt('stress_beck.txt', beck_stress_arr)
np.savetxt('pe_our.txt', pe_arr)
np.savetxt('pe_beck.txt', beck_pe_arr)

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])

plt.plot(data_dft[:,0], pe_arr, label='our work')
plt.plot(data_dft[:,0], beck_pe_arr, label='Beck1968')
plt.plot(data_dft[:,0], data_dft[:,2], label='DFT', color='black')
plt.xlabel('Lattice Constant/ A', fontsize=16)
plt.ylabel('Energy/ eV', fontsize=16)
plt.title('Energy-Lattice Constant curve of a HCP Helium Lattice', fontsize=16)
plt.legend( fontsize=14)
plt.show()


pot = eam_fit.pot_lammps['He F']
plt.ylabel('Embedding Function / eV')
plt.xlabel('Electron Density/ pot_units')
plt.title('Embedding function of Helium')
plt.plot(np.linspace(0, eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho']), pot)
plt.show()


pot = eam_fit.pot_lammps['He-He p']
plt.ylabel('Electron Density / pot_units')
plt.xlabel('Distance/ A')
plt.title('Electron Density Function of Helium')
plt.plot(r, pot)
plt.show()


pot = eam_fit.pot_lammps['He-He'][1:]
pot /= r[1:]
plt.ylabel('Pairwise Potential / eV')
plt.xlabel('Distance/ A')
plt.title('He-He Pairwise Potential')
plt.plot(r[201:], pot[200:])
print(pot.min())
plt.show()

