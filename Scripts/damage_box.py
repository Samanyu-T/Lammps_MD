import sys
import os
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

output_folder = 'Test_Damage'

if me == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))

comm.barrier()

init_dict['size'] = 60

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

depth = np.linspace(0, 10, 10)

lmp_class = LammpsParentClass(init_dict, comm, me)

lmp = lammps()# cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

lmp_class.init_from_box(lmp)

lmp.command('dump mydump all custom 100 %s/test_damage.*.atom id type x y z' % os.path.join(output_folder,'Atom_Files'))
lmp.command('run 0')

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

nvacs = int(1e-1 * natoms)

vac_pos = None
if me == 0:
    vac_pos = lims_width*np.hstack([np.random.rand(nvacs, 2), np.random.beta(a=2, b=1.25, size=(nvacs, 1))])
comm.barrier()
vac_pos = comm.bcast(vac_pos, 0)

# find atoms nearest to cascade centres to apply kicks to
vac_idx = np.array([xf_ktree.query(cmati@_cpos, k=1)[1] for _cpos in vac_pos]) # +1 for LAMMPS

vac_idx = np.unique(vac_idx)

cmd_string = ''

for _idx in vac_idx:
    _idx += 1
    cmd_string += '%d ' % _idx 

lmp.command('group gvac id %s' % cmd_string)

lmp.command('delete_atoms group gvac compress yes')

vac_pos_actual = _x[vac_idx] + np.array([xlo,ylo,zlo])

int_pos = None
if me == 0:
    int_pos = lims_width*np.hstack([np.random.rand(nvacs, 2), np.random.beta(a=2, b=1.25, size=(nvacs, 1))])
comm.barrier()
int_pos = comm.bcast(int_pos, 0)

int_idx = np.array([xf_ktree.query(cmati@_cpos, k=1)[1] for _cpos in int_pos]) + 1 # +1 for LAMMPS

int_idx = np.unique(int_idx)

int_pos = _x[int_idx] + 0.25 * (2 * np.random.randint(low=0, high=2, size=(len(int_idx), 3)) - 1)

int_pos = int_pos % lims_width

for _pos in int_pos:
    lmp.command('create_atoms 1 single %f %f %f units box' % (_pos[0], _pos[1], _pos[2]))


lmp_class.cg_min(lmp)

he_frac = 1e-1 * natoms * 1e-2

h_frac = 2e-1 * natoms * 1e-2

rng = None
if me == 0:
    rng = np.random.randint(low=0, high = 100000)
comm.barrier()
rng = comm.bcast(rng,0)
lmp.command('create_atoms 2 random %d %d NULL overlap 1.0 maxtry 50' % (h_frac, rng))

rng = None
if me == 0:
    rng = np.random.randint(low=0, high = 100000)
comm.barrier()
rng = comm.bcast(rng,0)
lmp.command('create_atoms 3 random %d %d NULL overlap 1.0 maxtry 50' % (he_frac, rng))

lmp.command('write_data slice.restart')