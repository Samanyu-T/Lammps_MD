import sys
import os
import json
import numpy as np
from mpi4py import MPI
import itertools
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def get_tetrahedral_sites():

    tet_sites_0 = np.zeros((12,3))
    k = 0

    for [i,j] in itertools.combinations([0, 1, 2],2):
        tet_sites_0[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                            [0.25, 0.5],
                                            [0.5 , 0.75],
                                            [0.75, 0.5] ])

        k += 1

    tet_sites_1 = np.ones((12,3))
    k = 0

    for [i,j] in itertools.combinations([0, 1, 2],2):
        tet_sites_1[4*k:4*k+4,[i,j]] = np.array( [[0.5 , 0.25],
                                            [0.25, 0.5],
                                            [0.5 , 0.75],
                                            [0.75, 0.5] ])

        k += 1

    tet_sites_unit = np.vstack([tet_sites_0, tet_sites_1])

    # n_iter = np.clip(self.n_vac, a_min = 1, a_max = None)

    # tet_sites = np.vstack([tet_sites_unit + i*0.5 for i in range(n_iter)])

    return np.unique(tet_sites_unit, axis = 0)

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Lammps_Files_Helium_Testing'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 4

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

lmp = LammpsParentClass(init_dict, comm, proc_id)

lmp.perfect_crystal()

tet_unit = get_tetrahedral_sites()

tet_stack = np.vstack([tet_unit + [1,0,0], tet_unit + [0,1,0], tet_unit + [0,0,1]])

tet = np.unique(tet_stack, axis = 0)

pe_lst = []

sites = []

pos0 = np.array([0.25, 0.5, 0])

pos100 = lmp.alattice*np.array([0.5, 0, 0])

pe, rvol, xyz = lmp.create_atoms_given_pos(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'), os.path.join(output_folder, 'Data_Files', 'He_test100.data' ),
                                [3], [pos100])
print(pe)
exit()
for i, _tet in enumerate(tet):

    if not np.all(_tet == pos0):
        pos = lmp.alattice*(np.row_stack([pos0, _tet]) + lmp.size//2)


        pe, rvol, xyz = lmp.create_atoms_given_pos(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'), os.path.join(output_folder, 'Data_Files', 'He_test_%d.data' % i),
                                [3, 3], pos)
        pe = round(pe, 1)
        pe_lst.append(pe)
        sites.append(_tet)
        # print(pe, rvol, (xyz % lmp.alattice)/lmp.alattice)
pe_lst = np.array(pe_lst)
pe_unique, idx_unique = np.unique(pe_lst, return_index=True)

print(pe_unique, idx_unique)

for i, idx in enumerate(idx_unique):
    os.rename(os.path.join(output_folder, 'Atom_Files', 'He_test_%d.atom' % idx), os.path.join(output_folder, 'Atom_Files', 'He_vasp_%d.atom' % i))
exit()

# pos = lmp.alattice*(np.array([[0.25, 0.5, 0], [0, 0.5, 0.25]]) + 2)

# pe, rvol, xyz = lmp.create_atoms_given_pos(os.path.join(output_folder, 'Data_Files', 'V0H0He0.data'), os.path.join(output_folder, 'Data_Files', 'He_test_1.data'),
#                            [3, 3], pos)

print(pe, rvol, (xyz % lmp.alattice)/lmp.alattice)

init_path =  os.path.join(output_folder, 'Data_Files', 'He_test_0.data')
potfile = lmp.potfile
final_path =  os.path.join(output_folder, 'Atom_Files', 'He_test_1.atom')
neb_final_path =  os.path.join(output_folder, 'Atom_Files', 'He_test_1.neb')

Natoms = 2*lmp.size**3 + 2
data = np.loadtxt(final_path, skiprows=9)

with open(neb_final_path, 'w') as file:
    file.write('%d\n' % Natoms)
    for _data in data:
        file.write('%d %.4f %.4f %.4f\n' % (_data[0], _data[2], _data[3], _data[4]))

txt = '''
units metal 

atom_style atomic

atom_modify map array sort 0 0.0

read_data %s

mass 1 183.84

mass 2 1.00784

mass 3 4.002602

pair_style eam/alloy

pair_coeff * * %s W H He

thermo 10

run 0

fix 1 all neb 1e-4

timestep 1e-3

min_style quickmin

thermo 100 

variable i equal part

neb 10e-8 10e-10 5000 5000 100 final %s

write_dump all custom %s/neb.$i.atom id type x y z ''' % (init_path, potfile, neb_final_path, output_folder)

with open('he.neb', 'w') as file:
    file.write(txt)

