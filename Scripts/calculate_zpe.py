import sys
import os
import json
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from lammps import lammps

def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

# comm = 0

# proc_id = 0

# n_procs = 1

init_dict = {}

with open('init_param.json', 'r') as file:
    init_dict = json.load(file)

output_folder = 'SIA_Binding'


if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder,'Data_Files'))
    os.mkdir(os.path.join(output_folder,'Atom_Files'))

def eval_energy(x, lmp, lmp_class, N):

    lmp.command('create_atoms 3 single %f %f %f units box' % (x[0], x[1], x[2]))
    
    lmp.command('run 0')

    _x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)

    ef = lmp_class.get_formation_energy(lmp, N_species=[2*lmp_class.size**3, 0, 1])

    force = np.ctypeslib.as_array(lmp.gather_atoms("f", 1, 3)).reshape(N, 3)

    # lmp.command('write_data test.data')

    # print(_x[-1], ef, force[-1])

    lmp.command('group helium type 3')

    lmp.command('delete_atoms group helium')

    return ef, force[-1]

def calc_hessian_energy(x0):
    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp.command('create_atoms 3 single %f %f %f units lattice' % (x0[0], x0[1], x0[2]))
    
    N = lmp.get_natoms()

    _x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)

    x0 = _x[-1]

    lmp_class.cg_min(lmp)
    
    ef = lmp_class.get_formation_energy(lmp, N_species=[2*lmp_class.size**3, 0, 1])

    print(ef, x0)

    lmp.command('group helium type 3')

    lmp.command('delete_atoms group helium')

    n = 3
    hessian = np.zeros((n, n))

    h = 1e-3

    for i in range(n):
        for j in range(n):
            # Create unit vectors
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = 1
            ej[j] = 1
            
            # Calculate each term in the finite difference formula
            term1, force1 = eval_energy(x0 + h * ei + h * ej, lmp, lmp_class, N,)
            term2, force2 = eval_energy(x0 - h * ei + h * ej, lmp, lmp_class, N,)
            term3, force3 = eval_energy(x0 + h * ei - h * ej, lmp, lmp_class, N,)
            term4, force4 = eval_energy(x0 - h * ei - h * ej, lmp, lmp_class, N,)
            
            # Calculate the Hessian element H[i, j]
            hessian[i, j] = (term1 - term2 - term3 + term4) / (4 * h**2)

    lmp.close()

    return hessian


def calc_hessian_force(x0):
    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp.command('create_atoms 3 single %f %f %f units lattice' % (x0[0], x0[1], x0[2]))
    
    N = lmp.get_natoms()

    _x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(N, 3)

    x0 = _x[-1]

    lmp_class.cg_min(lmp)
    
    ef = lmp_class.get_formation_energy(lmp, N_species=[2*lmp_class.size**3, 0, 1])

    print(ef, x0)

    lmp.command('group helium type 3')

    lmp.command('delete_atoms group helium')

    n = 3  # Dimension of the input vector
    hessian = np.zeros((n, n))  # Initialize Hessian matrix
    h = 1e-1

    for j in range(n):
        # Create unit vector for direction j
        ej = np.zeros(n)
        ej[j] = 1
        
        # Calculate the gradient at (x + h*ej) and (x - h*ej)
        ef_plus, force_plus = eval_energy(x0 + h * ej, lmp, lmp_class, N)  # g_i(x + h * e_j)
        ef_minus, force_minus = eval_energy(x0 - h * ej, lmp, lmp_class, N)  # g_i(x - h * e_j)

        # Calculate each entry of the Hessian
        for i in range(n):
            hessian[i, j] = -(force_plus[i] - force_minus[i]) / (2 * h)

    lmp.close()

    return hessian


def calc_hessian(x0, filename):
    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp.command('region rinterest sphere %f %f %f %f units lattice' % (3.5, 3.5, 3, 1))
    
    lmp.command('group ginterest region rinterest')

    if x0 is not None:
        lmp.command('create_atoms 3 single %f %f %f units lattice' % (x0[0], x0[1], x0[2]))

        lmp.command('group helium type 3')

        lmp.command('group dynamic union ginterest helium')

        lmp_class.cg_min(lmp)

        pe = lmp.get_thermo('pe')

        print(pe)

        lmp.command('dynamical_matrix dynamic eskm 1e-4 file %s' % filename)

    else:

        lmp_class.cg_min(lmp)

        pe = lmp.get_thermo('pe')

        print(pe)

        lmp.command('dynamical_matrix ginterest eskm 1e-4 file %s' % filename)

    lmp.close()

    line_prepender(filename, '%f' % pe)


init_dict['size'] = 6

init_dict['surface'] = 0

init_dict['output_folder'] = output_folder

init_dict['orientx'] = [1, 0, 0]

init_dict['orienty'] = [0, 1, 0]

init_dict['orientz'] = [0, 0, 1]

# init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.he'

init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

init_dict['pottype'] = 'he'

# init_dict['potfile'] = 'git_folder/Potentials/beck.eam.alloy'

# init_dict['pottype'] = 'alloy'

x0 = np.array([3.25, 3.5, 3])

hessian = calc_hessian(x0, 'tet_hessian.dat')


x0 = np.array([3.5, 3.5, 3])

hessian = calc_hessian(x0, 'oct_hessian.dat')


x0 = np.array([3.4, 3.4, 3])

hessian = calc_hessian(x0, 'tri_hessian.dat')



hessian = calc_hessian(None, 'perfect_hessian.dat')