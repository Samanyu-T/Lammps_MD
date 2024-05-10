import sys
import os
import numpy as np
import json
from mpi4py import MPI
from glob import glob

sys.path.append(os.path.join(os.getcwd(), 'git_folder','Classes'))

from Lammps_Classes import Monte_Carlo_Methods

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_Files_7x7_MC_New'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 7

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = Monte_Carlo_Methods(init_dict, comm, proc_id)
lmp.perfect_crystal()

struct_lst = []
pe_lst = []
rvol_lst = []
xyz_lst = []

p_events_dict = {'displace':0.75, 'exchange':0.25, 'delete':0, 'create':0}

for file in glob('Lammps_Files_7x7/Data_Files/*'):
    
    filename = os.path.basename(file).split('.')[0]

    v = int(filename[1])

    h = int(filename[3])

    he = int(filename[-1])
    
    pe_arr, rvol_arr, xyz_accept_lst, n_species, ratio = lmp.monte_carlo(file, [2,3], 500, p_events_dict,
                                              temp = 100, potential = 0, max_displacement = 0.5*np.ones((3,)),
                                              region_of_interest= None, save_xyz=False, diag = False )
   

    pe_sort_idx = np.argsort(pe_arr)

    N_slct = 5

    slct_idx = pe_arr.argmin() #[ int(i*(len(pe_sort_idx)-1)//(N_slct-1)) for i in range(N_slct)]

    struct_lst.append(np.array([v, h, he]))

    pe_lst.append(pe_arr[pe_sort_idx[slct_idx]])

    rvol_lst.append(rvol_arr[pe_sort_idx[slct_idx]])

    xyz_accept_lst = np.array(xyz_accept_lst)

    xyz_lst.append(xyz_accept_lst[pe_sort_idx[slct_idx]])

    print(file, pe_lst[-1], rvol_lst[-1], ratio)

struct_lst = np.array(struct_lst)
pe_lst = np.array(pe_lst)
rvol_lst = np.array(rvol_lst)

np.save('ef.npy', pe_lst)
np.save('rvol.npy', rvol_lst)
np.save('xyz.npy', struct_lst)


# np.savez('struct_mcmc_5x5.npz', *struct_lst)
# np.savez('ef_mcmc_5x5.npz', *pe_lst)
# np.savez('rvol_mcmc_5x5.npz', *rvol_lst)
# np.savez('config_mcmc_5x5.npz', *xyz_lst)