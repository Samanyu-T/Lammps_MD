import glob
from lammps import lammps
import sys, os, json
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from Lammps_Classes import LammpsParentClass
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_4x4'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 4

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

lmp_class = LammpsParentClass(init_dict, comm, proc_id)

data = []

for file in glob.glob('Lammps_Files_4x4/Data_Files/*.data'):
    filename = os.path.basename(file)

    vac = int(filename[1])

    h = int(filename[3])

    he = int(filename[6])
    
    lmp_class.N_species = np.array([2*lmp_class.size**3 - vac, h, he])

    lmp = lammps(comm=comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp.commands_list(lmp_class.init_from_datafile(file)) 

    lmp_class.cg_min(lmp)

    ef = lmp_class.get_formation_energy(lmp)

    rvol = lmp_class.get_rvol(lmp)

    lmp.close()

    _data =  [vac, h, he, ef, rvol]
            
    data.append(_data)

data = np.array(data)

np.savetxt('test_4x4_old_cg.txt', data)