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

def spherical_region(lmp, centre, r, nhe):

    lmp.command('region rsphere sphere %f %f %f %f units lattice' % (centre[0], centre[1], centre[2], r))

    lmp.command('delete_atoms region rsphere')

    if me == 0:
        rng = np.random.randint(low=1, high = 100000)

        lmp.command('create_atoms 3 random %d %d rsphere' % (nhe, rng))

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

temp = 300

for idx in range(100):

    nvac = None
    nhe = None
    r = None
    rmax = 0.1

    if me == 0:
        r = rmax * (0 * np.random.rand() + 0.4/rmax)
    
    comm.barrier()

    r = comm.bcast(r)

    lmp = lammps(cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)


    lmp.command('run 0')

    lmp_class.cg_min(lmp, 100000, True) 


    vol_perfect = lmp.get_thermo('vol')

    stress_perfect = get_stress(lmp)

    lmp_class.stress_perfect = stress_perfect

    lmp_class.vol_perfect = vol_perfect

    lmp_class.alattice = lmp.get_thermo('xlat')

    centre = [lmp_class.size//2  + 0.2, lmp_class.size//2 + 0.2, lmp_class.size//2 + 0.2]

    lmp.command('region rsphere sphere %f %f %f %f units lattice' % (centre[0], centre[1], centre[2], r))

    lmp.command('delete_atoms region rsphere')

    nvac = 2 * lmp_class.size ** 3 - lmp.get_natoms() 
    
    rng = None
    if me == 0:
        rng = np.random.randint(low=1, high = 100000)
        nhe = int(7 * nvac * np.random.rand())

    comm.barrier()

    rng = comm.bcast(rng)
    nhe = comm.bcast(nhe)
    
    lmp.command('create_atoms 2 random %d %d rsphere' % (nhe, rng))

    lmp_class.cg_min(lmp)
    
    # lmp.command('fix fnvt all nvt temp %f %d 100.0' % (temp, temp))

    lmp.command('fix fnve all nve')

    lmp.command('velocity all create %f %d rot no dist gaussian' % (2 * temp, rng))

    lmp.command('run 10000')
    
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
        print(rvol/nvac, nhe / nvac, nhe, nvac, (vol - vol_perfect)/vol_perfect, pressure)
    
    lmp.close()




    # lmp.command('compute        peratom all pe/atom')
    # lmp.command('compute        pe all reduce sum c_peratom')
    # lmp.command('thermo_style   custom step temp etotal press pe c_pe pxx pyy pzz pxy pxz pyz vol')

    # lmp.command('run 0')
    
    # vac_xyz = []

    # for i in range(nvac):
    #     _xyz = del_min_energy(lmp)
    #     vac_xyz.append(_xyz)

    # lmp_class.cg_min(lmp)

    # vac_xyz = np.array(vac_xyz)

    # he_xyz = np.zeros((nhe, 3))

    # if me == 0:
    #     for i in range(nhe):
    #         j = i % len(vac_xyz)
    #         he_xyz[i] = vac_xyz[j] + lmp_class.alattice * np.random.rand(3) * 0.75
    # comm.barrier()

    # he_xyz = comm.bcast(he_xyz, root = 0)

    # he_xyz = (he_xyz.flatten()).tolist()

    # type = [3 for i in range(nhe)]

    # _n = lmp.create_atoms(n=nhe, id = None, type = type, x = he_xyz, v=None, shrinkexceed=True)
    
    # print(_n, nhe)