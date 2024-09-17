import sys
import os, shutil
import json
import numpy as np
from mpi4py import MPI
from lammps import lammps
from scipy.spatial import cKDTree
import copy
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

comm = MPI.COMM_WORLD

me = comm.Get_rank()

n_procs = comm.Get_size()

# comm = 0

# me = 0

# n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Dislocation_Loops'


if me == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))

comm.barrier()

init_dict['size'] = 12

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder


init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

# init_dict['orientx'] = [1, 1, 1]
# init_dict['orienty'] = [-1,2,-1]
# init_dict['orientz'] = [-1,0, 1]

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, me)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp_class.init_from_box(lmp)

void_radius = 6 / init_dict['alattice']

centre = init_dict['size']/2

# extract cell dimensions 
natoms = lmp.extract_global("natoms", 0)
xlo, xhi = lmp.extract_global("boxxlo", 2), lmp.extract_global("boxxhi", 2)
ylo, yhi = lmp.extract_global("boxylo", 2), lmp.extract_global("boxyhi", 2)
zlo, zhi = lmp.extract_global("boxzlo", 2), lmp.extract_global("boxzhi", 2)
xy, yz, xz = lmp.extract_global("xy", 2), lmp.extract_global("yz", 2), lmp.extract_global("xz", 2)

# Relevant documentation: https://docs.lammps.org/Howto_triclinic.html 
xlb = xlo + min(0.0,xy,xz,xy+xz)
xhb = xhi + max(0.0,xy,xz,xy+xz)
ylb = ylo + min(0.0,yz)
yhb = yhi + max(0.0,yz)
zlb, zhb = zlo, zhi

lims_lower = np.r_[xlb, ylb, zlb]  # bounding box origin
lims_upper = np.r_[xhb, yhb, zhb]  # bounding box corner
lims_width = lims_upper-lims_lower # bounding box width

# Basis matrix for converting scaled -> cart coords
c1 = np.r_[xhi-xlo, 0., 0.]
c2 = np.r_[xy, yhi-ylo, 0.]
c3 = np.r_[xz, yz, zhi-zlo]
cmat =  np.c_[[c1,c2,c3]].T
cmati = np.linalg.inv(cmat)

_x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(natoms, 3)
_x = _x - np.array([xlo,ylo,zlo])

_xfrac = np.einsum("ij,kj->ki", cmati, _x)

# build KDTree (in fractional coords) for nearest neighbour search containing all atomic data
xf_ktree = cKDTree(_xfrac, boxsize=[1,1,1])

centre = lims_width/2 

# int_pos = None
# N = 10
# if me == 0:
#     r = 3* np.random.rand(N)
#     theta = 2 * np.pi * np.linspace(0, 1, N)
#     int_pos = np.column_stack([  r * np.cos(theta), r * np.sin(theta), np.zeros((N,)) ]) + centre

# comm.barrier()
# int_pos = comm.bcast(int_pos, 0)

# lmp.create_atoms(n=N, id = None, type = np.ones((N,),dtype=int).tolist(), x = (int_pos.flatten()).tolist(), v=None)

basis = np.array([[0,0,1], [1,0,1], [1,0,0], [1,1,0], [0,1,0], [0,1,1]])
n = 3
# +z, +x, -z, +y, -x, +z, -y
vec = [[0, 0, 0]]
for i in range(n):
    s = [0, 0, i + 1]

    for j in range(i + 1):
        k = j + 1
        _s = [s[0] + k, s[1], s[2]]
        vec.append(_s)

    s = copy.copy(_s)

    for j in range(i + 1):
        k = j + 1
        _s = [s[0], s[1], s[2] - k]
        vec.append(_s)

    s = copy.copy(_s)

    for j in range(i + 1):
        k = j + 1
        _s = [s[0], s[1] + k, s[2]]
        vec.append(_s)

    s = copy.copy(_s)
    for j in range(i + 1):
        k = j + 1
        _s = [s[0] - k, s[1], s[2]]
        vec.append(_s)

    s = copy.copy(_s)
    for j in range(i + 1):
        k = j + 1
        _s = [s[0], s[1], s[2] + k]
        vec.append(_s)

    s = copy.copy(_s)
    for j in range(i + 1):
        k = j + 1
        _s = [s[0], s[1] - k, s[2]]
        vec.append(_s)

    s = copy.copy(_s)

vec = np.array(vec)
vec = np.unique(vec, axis = 0)
vec = vec + 0.25 + 5
# vec = vec * 3.14221
# vec = vec % lims_width
# vec += vec + np.array([xlo, ylo, zlo])
N = int(len(vec))
x_list = [float(_x) for _x in vec.flatten()]
print(vec)
# lmp.create_atoms(n=N, id = None, type = [int(1) for i in range(N)], x = x_list)
for i in range(N):
    lmp.command('create_atoms 1 single %f %f %f units lattice' % (vec[i, 0], vec[i, 1], vec[i, 2]))

lmp.command('run 0')
lmp.command('dump mydump all custom 1000 %s/interstitial_loop.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))
lmp_class.cg_min(lmp)

pe_0 = lmp.get_thermo('pe')
ef0 = lmp_class.get_formation_energy(lmp, np.array([2*lmp_class.size**3 + N, 0 , 0]))

N_he = 1

vec_he = np.zeros((N_he, 3))

for i in range(N_he):
    # randint = np.random.randint(low = N-10, high = N)
    vec_he[i] = lims_width * np.random.rand(3) + np.array([xlo, ylo, zlo])

# lmp.command('write_data %s/sia_loop.data' % os.path.join(output_folder,'Data_Files'))

# input_filepath = '%s/sia_loop.data' % os.path.join(output_folder,'Data_Files')
# output_filepath = '%s/sia_loop_he.data' % os.path.join(output_folder,'Data_Files')

# defect_centre = 2.25 * 3.14* np.array([[1,1,1]])
# ef, rvol = lmp_class.add_defect(input_filepath, output_filepath, 3, 1, defect_centre, minimizer='random', run_MD=True)

lmp.command('create_atoms 3 single %f %f %f units box' % (6.25*3.14221, 6.5*3.14221, 4*3.14221))

lmp_class.run_MD(lmp, temp=600, timestep=1e-3, N_steps= 10000)

lmp_class.cg_min(lmp)

output_filepath = '%s/sia_loop_he.data' % os.path.join(output_folder,'Data_Files')

lmp.command('write_data %s' % output_filepath)
pe_1 = lmp.get_thermo('pe')

print(pe_0 + 6.73 - pe_1)