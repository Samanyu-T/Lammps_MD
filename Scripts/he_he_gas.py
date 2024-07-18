import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
import matplotlib.pyplot as plt

potfile =  'git_folder/Potentials/WHHe_test.eam.alloy' #'Fitting_Runtime/Potentials/optim.0.eam.alloy' 

lat_arr = np.linspace(2.5, 4, 20)
pe_arr = np.zeros(lat_arr.shape)
stress_arr = np.zeros(lat_arr.shape)
vol_arr = np.zeros(lat_arr.shape)

for i, lat in enumerate(lat_arr):
    lmp = lammps()
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
    lmp.command('lattice hcp %f' % lat)
    lmp.command('region r_simbox block 0 4 0 4 0 4 units lattice')
    lmp.command('region r_atombox block 0 4 0 4 0 4 units lattice')
    lmp.command('create_box 3 r_simbox')
    lmp.command('create_atoms 3 region r_atombox')
    lmp.command('mass 1 183.84')
    lmp.command('mass 2 1.00784')
    lmp.command('mass 3 4.002602')
    lmp.command('pair_style eam/alloy' )
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    lmp.command('minimize 1e-9 1e-12 10 10')
    lmp.command('minimize 1e-12 1e-15 100 100')
    lmp.command('minimize 1e-16 1e-18 %d %d' % (1000, 1000))

    hydrostatic_pressure = (lmp.get_thermo('pxx') +  lmp.get_thermo('pyy') +  lmp.get_thermo('pzz') )/3
    volume = lmp.get_thermo('vol') / 256
    pe_arr[i] = lmp.get_thermo('pe')
    stress_arr[i] = hydrostatic_pressure
    vol_arr[i] = volume


plt.plot(vol_arr, stress_arr)
plt.xlabel('Atomic Volume/ A**3')
plt.ylabel('Hydrostatic Stress / bar')
plt.title('Pressure-Volume curve of a HCP Helium Lattice')
plt.show()


plt.plot(lat_arr, pe_arr)
plt.show()