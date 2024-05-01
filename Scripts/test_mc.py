import sys
import os
import numpy as np
from glob import glob

sys.path.append(os.path.join(os.getcwd(), 'Classes'))

from Lammps_Classes import OptimizeStructures

lmp = OptimizeStructures('init_param.json')
lmp.perfect_crystal()


for file in glob('Lammps_Files/Data_Files/V0H9He9*'):
    
    filename = os.path.basename(file).split('.')[0]

    x = int(filename[1])

    y = int(filename[3])

    z = int(filename[-1])
    
    pe_arr, rvol_arr, ratio = lmp.monte_carlo(file, species_of_interest=[2, 3], N_steps=100, p_exchange=0.2, p_remove=0.2, potential=1, diag=True)

    print(pe_arr, ratio)