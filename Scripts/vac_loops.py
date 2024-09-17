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

output_folder = 'Dislocation_Loops'


if me == 0:
    shutil.rmtree(output_folder)
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

vac_pos = None
loop_rad = 10
if me == 0:
    N = 2500
    r = loop_rad * np.sqrt(np.random.rand(N))
    theta = np.random.rand(N) * 2 * np.pi
    vac_pos = np.column_stack([  r * np.cos(theta), r * np.sin(theta), np.zeros((N,)),  ]) + centre

comm.barrier()
vac_pos = comm.bcast(vac_pos, 0)

# find atoms nearest to cascade centres to apply kicks to
vac_idx = np.array([xf_ktree.query(cmati@_cpos, k=1)[1] for _cpos in vac_pos]) # +1 for LAMMPS

vac_idx = np.unique(vac_idx)

cmd_string = ''

for _idx in vac_idx:
    _idx += 1
    cmd_string += '%d ' % _idx 

# lmp_class.cg_min(lmp, fix_aniso=True) 


lmp.command('dump mydump all custom 1000 %s/vac_loop.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))

lmp.command('run 0')

lmp.command('group gvac id %s' % cmd_string)

lmp.command('delete_atoms group gvac compress yes')

# lmp_class.cg_min(lmp)

lmp_class.run_MD(lmp, 1200, 1e-3, 10000)

lmp_class.cg_min(lmp)

# pe0 = lmp.get_thermo('pe')

# n_he = 20 #int(np.ceil(natoms * 1e-2 * 0.5))
# n_h =  20 #int(np.ceil(natoms * 1e-2 * 0.5))

# rng=None
# if me == 0:
#     rng = np.random.randint(low = 1, high = 100000)
# comm.barrier()
# rng = comm.bcast(rng, 0)

# lmp.command('region dislocation sphere %f %f %f %f side in units box' % (centre[0], centre[1], centre[2], loop_rad + 5))

# lmp.command('create_atoms 3 random %d %d dislocation' % (n_he, rng))


# rng=None
# if me == 0:
#     rng = np.random.randint(low = 1, high = 100000)
# comm.barrier()
# rng = comm.bcast(rng, 0)
# lmp.command('create_atoms 2 random %d %d dislocation' % (n_h, rng))

# lmp_class.run_MD(lmp, 400, 1e-4, 10000)

# lmp_class.cg_min(lmp)

# pe1 = lmp.get_thermo('pe')

# print(pe0 + n_he*6.73 + n_h*0.798 - pe1)