import sys
import os
import json
import numpy as np
from lammps import lammps
from mpi4py import MPI

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

comm = MPI.COMM_WORLD

me = comm.Get_rank()

nprocs = comm.Get_size()

def get_stress(lmp):
    pxx = lmp.get_thermo('pxx')
    pyy = lmp.get_thermo('pyy')
    pzz = lmp.get_thermo('pzz')
    pxy = lmp.get_thermo('pxy')
    pxz = lmp.get_thermo('pxz')
    pyz = lmp.get_thermo('pyz')

    return np.array([pxx, pyy, pzz, pxy, pxz, pyz])

def del_min_energy(lmp):
    lmp.command('run 0')

    pe_lcl = lmp.numpy.extract_compute('peratom', 1, 1)
    id_lcl = lmp.numpy.extract_atom('id', 0)
    xyz_lcl = lmp.numpy.extract_atom('x', 3)

    n_lcl = len(id_lcl)

    comm.barrier()

    pe_gbl = comm.gather(pe_lcl, root = 0)
    id_gbl = comm.gather(id_lcl, root = 0)
    xyz_gbl = comm.gather(xyz_lcl, root = 0)

    del_id = None
    del_xyz = None

    if me == 0:
        # Concatenate data if needed
        pe_gbl = np.concatenate(pe_gbl)  # Concatenate potential energies
        id_gbl = np.concatenate(id_gbl)  # Concatenate IDs
        xyz_gbl = np.concatenate(xyz_gbl, axis=0)  # Concatenate coordinates

        del_id = id_gbl[np.argmax(pe_gbl)] 
        del_xyz = xyz_gbl[np.argmax(pe_gbl)] 

    comm.barrier()
    del_id = comm.bcast(del_id, root=0)
    del_xyz = comm.bcast(del_xyz, root=0)
    lmp.command('group g_del id %d' % del_id)
    lmp.command('delete_atoms group g_del')

    return del_xyz

def spherical_region(lmp, centre, r, nhe, temp = 300):

    lmp.command('region rsphere sphere %f %f %f %f units lattice' % (centre[0], centre[1], centre[2], r))

    lmp.command('delete_atoms region rsphere')

    if me == 0:
        rng = np.random.randint(low=1, high = 100000)

        lmp.command('create_atoms 3 random %d %d rsphere' % (nhe, rng))

def calc_rvol(r, occ, type, temp = 300, idx = 0):

    lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp.command('run 0')

    lmp_class.cg_min(lmp, 100000, True) 

    vol_perfect = lmp.get_thermo('vol')

    stress_perfect = get_stress(lmp)

    lmp_class.stress_perfect = stress_perfect

    lmp_class.vol_perfect = vol_perfect

    lmp_class.alattice = lmp.get_thermo('xlat')

    centre = [lmp_class.size//2 + 0.2, lmp_class.size//2 + 0.2, lmp_class.size//2 + 0.2]

    lmp.command('region rsphere sphere %f %f %f %f units lattice' % (centre[0], centre[1], centre[2], r))

    lmp.command('delete_atoms region rsphere')

    nvac = 2 * lmp_class.size ** 3 - lmp.get_natoms() 
    
    ngas = int(occ * nvac)

    rng = None
    if me == 0:
        rng = np.random.randint(low=1, high = 100000)
    comm.barrier()

    rng = comm.bcast(rng)
    
    lmp.command('create_atoms %d random %d %d rsphere' % (type, ngas, rng))

    lmp_class.cg_min(lmp)
    
    # lmp.command('fix fnvt all nvt temp %f %d 100.0' % (temp, temp))

    lmp.command('fix fnve all nve')

    lmp.command('velocity all create %f %d rot no dist gaussian' % (2 * temp, rng))

    lmp.command('run 5000')
    
    # lmp_class.cg_min(lmp, conv = 100000)

    # lmp.command('velocity all create %f %d rot no dist gaussian' % (temp, rng))

    # lmp.command('run 5000')
    
    lmp_class.cg_min(lmp, conv = 100000)

    lmp.command('velocity all create %f %d rot no dist gaussian' % (1e-3, rng))

    lmp.command('run 100')

    lmp_class.cg_min(lmp, conv = 100000, fix_aniso=True)

    lmp.command('write_dump all custom %s id type x y z'  % os.path.join(output_folder, 'test.%d.atom' % idx))

    vol = lmp.get_thermo('vol')
    

    # Find the Voigt Stress
    pxx = lmp.get_thermo('pxx')
    pyy = lmp.get_thermo('pyy')
    pzz = lmp.get_thermo('pzz')
    pxy = lmp.get_thermo('pxy')
    pxz = lmp.get_thermo('pxz')
    pyz = lmp.get_thermo('pyz')
    
    # vol = lmp.get_thermo('vol')
    # stress_voigt = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) - stress_perfect

    # strain_tensor = lmp_class.find_strain(stress_voigt)
    
    # delta_vol = np.trace(strain_tensor)*vol_perfect + (vol - vol_perfect)

    # rvol = 2 * delta_vol / (lmp.get_thermo('xlat') * lmp.get_thermo('ylat') * lmp.get_thermo('zlat'))

    pressure = lmp.get_thermo('press')

    rvol = lmp_class.get_rvol(lmp)

    if me == 0:
        print(rvol/nvac, ngas / nvac, ngas, nvac, (vol - vol_perfect)/vol_perfect, pressure)
    
    lmp.close()


init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'Rvol_Gas_Clusters'

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

init_dict['size'] = 16

init_dict['surface'] = 0

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

init_dict['output_folder'] = output_folder

lmp_class = LammpsParentClass(init_dict, comm)

if me == 0:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
comm.barrier()


occ = np.hstack([#np.linspace(0, 6, 7), np.linspace(0, 6, 12), np.linspace(0, 6, 15), 
                 np.linspace(0, 6, 18)#, np.linspace(0, 6, 20)

               ])

r = np.hstack([#0.4 * np.ones((7, )), 0.6 * np.ones((12, )), 0.82 * np.ones((15, )),
                1.0 * np.ones((18, ))#, 1.25 * np.ones((20, ))
              ])

idx = 0 

for _n, _r in zip(occ, r):
    calc_rvol(_r, _n, 2, 100, idx)
    idx += 1

# temp = 300

# for _i in range(100):

#     rmax = 1
#     r = None
#     occ = None

#     if me == 0:
#         r = rmax * (np.random.rand() + 0.4/rmax)
#         occ = np.random.randint(low = 0, high = 7)

    
#     comm.barrier()

#     r = comm.bcast(r)
#     occ = comm.bcast(occ)

#     calc_rvol(r, occ, 2, 300, idx)

#     idx += 1