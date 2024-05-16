import sys
import os
import json
import numpy as np
from mpi4py import MPI

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_Dislocations'


# init_dict['orientx'] = [1, 0, 0]

# init_dict['orienty'] = [0, 1, 0]

# init_dict['orientz'] = [0, 0, 1]



# init_dict['orientx'] = [1, 0, 0]

# init_dict['orienty'] = [0, 1, 0]

# init_dict['orientz'] = [0, 0, 1]



init_dict['orientx'] = [1, 1, -1]

init_dict['orienty'] = [-1, 1, 0]

init_dict['orientz'] = [1, 1, 2]


init_dict['size'] = 16

init_dict['surface'] = 0

init_dict['conv'] = 10000

init_dict['output_folder'] = output_folder

fix_cmd = []

norm = 8

fix_layers = 10

fix_cmd.append('variable radius atom (x^%d+y^%d)^(1/%d)' % (norm, norm, norm))

fix_cmd.append('variable select atom "v_radius  > %f" ' % (init_dict['alattice']*(init_dict['size'] - fix_layers)))

fix_cmd.append('group fixpos variable select')

fix_cmd.append('fix freeze fixpos setforce 0.0 0.0 0.0')

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = LammpsParentClass(init_dict, comm, proc_id)

pe_edge = lmp.generate_edge_dislocation('Edge',b=init_dict['alattice'], fix_cmd=fix_cmd)

pe, _ = lmp.add_defect(os.path.join(output_folder, 'Data_Files', 'Edge.data'), os.path.join(output_folder, 'Data_Files', 'He_Edge.data'), 3, 1, np.array([0, 0, lmp.alattice*lmp.size/2]), fix_cmd)

print(pe - pe_edge - 6.12)


lmp.orientx = [1, -1, 0]

lmp.orienty  = [1, 1, -2]

lmp.orientz = [1, 1, 1]

lmp.size = 10

norm = 2

fix_layers = 4

fix_cmd = []

fix_cmd.append('variable radius atom (x^%d+y^%d)^(1/%d)' % (norm, norm, norm))

fix_cmd.append('variable select atom "v_radius  > %f" ' % (init_dict['alattice']*(init_dict['size'] - fix_layers)))

fix_cmd.append('group fixpos variable select')

fix_cmd.append('fix freeze fixpos setforce 0.0 0.0 0.0')

pe_screw = lmp.generate_screw_dislocation('Screw', b=init_dict['alattice'], fix_cmd=fix_cmd)

pe, _ = lmp.add_defect(os.path.join(output_folder, 'Data_Files', 'Screw.data'), os.path.join(output_folder, 'Data_Files', 'He_Screw.data'), 3, 1, np.array([0, 0, lmp.alattice*lmp.size/2]), fix_cmd)

print(pe - pe_screw - 6.12)
