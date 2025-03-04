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

def hydrogen_helium(r, potfile, type='he'):
    lmp = lammps()
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary f f f')
    lmp.command('region r_simbox block 0 10 0 10 0 10 units box')     
    lmp.command('create_box 3 r_simbox')
    lmp.command('mass 1 183.84')
    lmp.command('mass 2 1.00784')
    lmp.command('mass 3 4.002602')
    lmp.command('pair_style eam/%s' % type)
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('create_atoms 2 single 0 0 0')
    lmp.command('create_atoms 3 single 0 0 %f' % r)
    lmp.command('run 0')
    xyz = np.array(lmp.gather_atoms('x', 1, 3)).reshape(2, 3)
    print(xyz)
    pe = lmp.get_thermo('pe')
    lmp.close()
    return pe
N = 100

mnl = np.zeros((N, ))

r = np.linspace(1, 4, N)

for i, _r in enumerate(r):
    mnl[i] = hydrogen_helium(r = _r, potfile= 'git_folder/Potentials/mnl-tb.eam.he', type='he')

plt.plot(r, mnl, label='Our Work')


beck = np.zeros((N, ))

r = np.linspace(1, 4, N)

for i, _r in enumerate(r):
    beck[i] = hydrogen_helium(r = _r, potfile= 'git_folder/Potentials/beck_full.eam.he', type='he')

plt.plot(r, beck, label='LJ - from paper')


xcli = np.zeros((N, ))

r = np.linspace(1, 4, N)

for i, _r in enumerate(r):
    xcli[i] = hydrogen_helium(r = _r, potfile= 'git_folder/Potentials/xcli.eam.fs', type='fs')

plt.plot(r, xcli, label='XC Li')


data_dft = []

with open('hhe_energy.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        r =  float(split[0][-3:])
        pe = float(split[-1])
        data_dft.append([r, pe])
data_dft = np.array(data_dft)

plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')

np.savetxt('hhe-mnl.txt', mnl)
np.savetxt('hhe_beck.txt', mnl)
np.savetxt('hhe_xcli.txt', mnl)

# pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/mnl-tb.eam.he')

# r = np.linspace(0, pot_params['rc'], pot_params['Nr'])[1:]
# hhe = pot['H-He'][1:]
# hhe = hhe / r
# plt.plot(r[400:], hhe[400:])

# plt.yscale('log')

plt.legend()

plt.show()