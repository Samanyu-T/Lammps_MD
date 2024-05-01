import sys
import os
import numpy as np
import json
from mpi4py import MPI

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import Monte_Carlo_Methods

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

comm_split = comm.Split(proc_id, proc_id)

potential_arr = np.linspace(0, 0.5, 2)

temp_arr = np.linspace(10, 910, 7)

replica_id_arr = np.arange(8)

potential = potential_arr[ proc_id % 2]

temp = temp_arr[ proc_id % 7]

replica_id = replica_id_arr[ proc_id % 8]

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)
 
output_folder = os.path.join('Monte_Carlo_HSurface', 'Pot_%.1f_Temp_%d_Replica_%d' % (potential, int(temp), int(replica_id)))

init_dict['size'] = 7

init_dict['orientx'] = [1, 1, 0]

init_dict['orienty'] = [0, 0, -1]

init_dict['orientz'] = [-1, 1, 0]

init_dict['surface'] = 20

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = Monte_Carlo_Methods(init_dict, comm_split, proc_id)

lmp.perfect_crystal()

print('work')
p_events_dict = {'displace':0.3, 'exchange':0, 'delete':0.1, 'create':0.6}

region_of_interest = np.vstack([lmp.offset, lmp.pbc + lmp.offset])

region_of_interest[: , -1] = np.array([0, -3])

print(region_of_interest)

pe_arr, rvol_arr, n_species_arr, xyz_accept_lst, ratio = lmp.monte_carlo(os.path.join(output_folder, 'Data_Files','V0H0He0.data'), [2], 1000, p_events_dict,
                                            temp = temp, potential = potential, max_displacement = 0.5*np.ones((3,)),
                                            region_of_interest= region_of_interest, save_xyz=False, diag = True )


np.savetxt(os.path.join(output_folder, 'pe.npy', pe_arr))
np.savetxt(os.path.join(output_folder, 'rvol.npy', pe_arr))
np.savetxt(os.path.join(output_folder, 'n_species.npy', n_species_arr))