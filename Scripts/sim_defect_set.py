import numpy as np
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import EAM_Fitting
import Handle_PotFiles
import time
from mpi4py import MPI
import json, glob, time
from Lammps_Classes import LammpsParentClass
from lammps import lammps

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()


init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'EAM_Fit_Files'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 4

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['potfile'] = 'git_folder/Potentials/test.0.eam.alloy'

t1 = time.perf_counter()

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

data = []

files = glob.glob('Fitting_Files/*.txt' )

for file in files:
    
    # V(1)H(3)He(6).(8).data
    filename = os.path.basename(file)

    vac = int(filename[1])

    h = int(filename[3])

    he = int(filename[6])

    image = int(filename[8])
    
    lmp_class.N_species = np.array([2*lmp_class.size**3 - vac, h, he])

    lmp = lammps(comm=comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.commands_list(lmp_class.init_from_box()) 

    # xyz = np.array(lmp.gather_atoms('x', 1, 3))

    # xyz = xyz.reshape(len(xyz)//3, 3)

    # print(filename)

    # print(xyz[-1]/lmp_class.alattice)

    if os.path.getsize(file) > 0:
        xyz = np.loadtxt(file)
    else:
        xyz = np.empty((0,3))

    if xyz.ndim == 1 and len(xyz) > 0:
        xyz = xyz.reshape(1, -1)

    for _v in range(vac):
        
        vac_pos = (2 - _v/2)*np.ones((3, ))

        lmp.command('region sphere_remove_%d sphere %f %f %f 0.1 units lattice' % 
                    (_v,vac_pos[0], vac_pos[1], vac_pos[2]))
        
        lmp.command('group del_atoms region sphere_remove_%d' % _v)

        lmp.command('delete_atoms group del_atoms')
        
        lmp.command('group del_atoms clear')
    
    if len(xyz) > 0:
        for _x in xyz:
            lmp.command('create_atoms %d single %f %f %f units lattice' % 
                        (_x[0], _x[1], _x[2], _x[3])
                        )

    lmp_class.cg_min(lmp)

    # lmp.command('write_dump all custom test_sim/V%dH%dHe%d.atom id type x y z' % (vac, h, he))

    ef = lmp_class.get_formation_energy(lmp)

    rvol = lmp_class.get_rvol(lmp)

    xyz = np.array(lmp.gather_atoms('x', 1, 3))

    xyz = xyz.reshape(len(xyz)//3, 3)

    _data =  [vac, h, he, image, ef, rvol]
    
    data.append(_data)

data = np.array(data)

sort_idx = np.lexsort((data[:, 3],data[:, 2], data[:, 1], data[:, 0]))

data = data[sort_idx]

t2 = time.perf_counter()

print(t2 - t1)
np.savetxt('sim.txt', data, fmt='%.3f')