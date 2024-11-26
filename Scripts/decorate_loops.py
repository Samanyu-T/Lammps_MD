import sys
import os, shutil
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps
from scipy.spatial import cKDTree

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

output_folder = 'Stable_Loops/sia_loop_111'


if me == 0:
    # shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))

comm.barrier()

init_dict['size'] = 10

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 1, 1]
init_dict['orienty'] = [-1,2,-1]
init_dict['orientz'] = [-1,0, 1]

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, me)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp.commands_list(lmp_class.init_from_datafile('Dislocation_Loops/Data_Files/sia_loop_he.data'))

# lmp_class.cg_min(lmp)


lmp.command('dump mydump all custom 1000 %s/sia_loop.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))

lmp.command('run 0')

pe0 = lmp.get_thermo('pe')

n_he = 20 #int(np.ceil(natoms * 1e-2 * 0.5))
n_h =  20 #int(np.ceil(natoms * 1e-2 * 0.5))

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

centre = lims_width/2

loop_rad = 10


_x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(natoms, 3)
_x = _x - np.array([xlo,ylo,zlo])

# _xfrac = np.einsum("ij,kj->ki", cmati, _x)

# build KDTree (in fractional coords) for nearest neighbour search containing all atomic data
xf_ktree = cKDTree(_x, boxsize=lims_width)

# h_pos = None
# if me == 0:
    
#     h_pos = []
#     for i in range(n_h):
#         while True:
#             r = loop_rad * np.sqrt(np.random.rand())
#             theta = np.random.rand() * 2 * np.pi
#             trial_pos = (np.array([  r * np.cos(theta), r * np.sin(theta), 5*np.random.rand()  ]) + centre) 
#             dist, _ = xf_ktree.query(trial_pos, k=1)
#             if dist > 1.25:
#                 h_pos.append(trial_pos)
#                 break
#     h_pos = np.array(h_pos)
# comm.barrier()

# h_pos = comm.bcast(h_pos, 0)
# h_pos = h_pos % lims_width
# comm.barrier()

# lmp.create_atoms(n=n_h, id = None, type = [2 for i in range(n_h)], x = (h_pos.flatten()).tolist(), v=None)

# he_pos = None
# if me == 0:
    
#     he_pos = []
#     for i in range(n_he):
#         while True:
#             r = loop_rad * np.sqrt(np.random.rand())
#             theta = np.random.rand() * 2 * np.pi
#             trial_pos = np.array([  r * np.cos(theta), r * np.sin(theta), 2*np.random.rand()  ]) + centre
#             dist, _ = xf_ktree.query(trial_pos, k=1)
#             if dist > 1.25:
#                 he_pos.append(trial_pos)
#                 break
#     he_pos = np.array(he_pos)
# comm.barrier()


# he_pos = comm.bcast(he_pos, 0)
# he_pos = he_pos % lims_width
# comm.barrier()

# lmp.create_atoms(n=n_he, id = None, type = [3 for i in range(n_h)], x = (he_pos.flatten()).tolist(), v=None)

# lmp_class.run_MD(lmp, 400, 5e-4, 10000)

# lmp_class.cg_min(lmp)

# print(centre - 10, centre + 10)
lmp.command('region disloc block %f %f %f %f %f %f units box' % (centre[0] - 10, centre[0] + 10, centre[1] - 10,
                                                          centre[1] + 10, centre[2] - 2.5, centre[2] + 2.5,))
rng=None
if me == 0:
    rng = np.random.randint(low = 1, high = 100000)
comm.barrier()
rng = comm.bcast(rng, 0)
lmp.command('create_atoms 2 random %d %d disloc overlap 1.5 maxtry 1000' % (n_h, rng))

rng=None
if me == 0:
    rng = np.random.randint(low = 1, high = 100000)
comm.barrier()
rng = comm.bcast(rng, 0)
lmp.command('create_atoms 3 random %d %d disloc overlap 1.5 maxtry 1000' % (n_he, rng))

lmp_class.run_MD(lmp, 1000, 5e-4, 1000)

lmp_class.cg_min(lmp)

pe1 = lmp.get_thermo('pe')

print(pe0 + n_he*6.73 + n_h*0.798 - pe1)