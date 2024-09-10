import sys
import os
import json
import numpy as np
from mpi4py import MPI
# import matplotlib.pyplot as plt
from lammps import lammps

sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))

from Lammps_Classes import LammpsParentClass

def gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder, natoms):

    neb_input_file = '''
units metal 
atom_style atomic
atom_modify map array sort 0 0.0
read_data %s
mass 1 183.84
mass 2 1.00784
mass 3 4.002602
pair_style eam/%s
pair_coeff * * %s W H He
thermo 10
run 0
# partition yes 1 fix freeze all setforce 0.0 0.0 0.0
# partition yes 7 fix freeze all setforce 0.0 0.0 0.0
fix 1 all neb 10000
timestep 1e-4
min_style quickmin
thermo 100 
variable i equal part
neb 10e-15 10e-18 50000 50000 10000 final %s
write_dump all custom %s/neb.$i.atom id type x y z
variable he_z equal z[%d]
print "He_z $(v_he_z:%s)" ''' %  (init_path, pottype, potfile, final_path, neb_image_folder, natoms,'%.6f')

    return neb_input_file

def main():

    comm = MPI.COMM_WORLD

    proc_id = comm.Get_rank()

    n_procs = comm.Get_size()

    print(proc_id, n_procs) 

    orient = sys.argv[1]

    print(orient)
    sys.stdout.flush()

    init_dict = {}

    with open('init_param.json', 'r') as file:
        init_dict = json.load(file)

    init_dict['size'] = 7

    init_dict['surface'] = 20

    if orient == 'orient100':
        init_dict['orientx'] = [1, 0, 0]
        init_dict['orienty'] = [0, 1, 0]
        init_dict['orientz'] = [0, 0, 1]
        init_dict['size'] = 12

    elif orient == 'orient110':
        init_dict['orientx'] = [1, 1, 0]
        init_dict['orienty'] = [0, 0, 1]
        init_dict['orientz'] = [1, -1, 0]

    elif orient == 'orient111':
        init_dict['orientx'] = [1, 1, 1]
        init_dict['orienty'] = [-1,2,-1]
        init_dict['orientz'] = [-1,0, 1]
    else:
        exit()
        
    # init_dict['potfile'] = 'Fitting_Runtime/Potentials/optim.0.eam.fs'

    init_dict['potfile'] = 'git_folder/Potentials/final.eam.he'

    init_dict['pottype'] = 'he'

    output_folder = '%s_neb' % orient

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder,'Data_Files'))
        os.mkdir(os.path.join(output_folder,'Atom_Files'))
        os.mkdir(os.path.join(output_folder,'Neb_Image_Folder'))

    init_dict['output_folder'] = output_folder

    potfile = init_dict['potfile']

    pottype = potfile.split('.')[-1]

    neb_image_folder = os.path.join(output_folder,'Neb_Image_Folder')

    lmp_class = LammpsParentClass(init_dict, comm, proc_id)

    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp_class.cg_min(lmp)

    pe_0 = lmp.get_thermo('pe')

    lmp.command('create_atoms 3 single %f %f %f units lattice' %  (3.2500, 3.500, 0.000))

    lmp_class.cg_min(lmp)

    pe_1 = lmp.get_thermo('pe')

    print(pe_1 - pe_0)

    lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'tet_1.data'))

    lmp.command('write_dump all custom %s id x y z'  % os.path.join(output_folder, 'Atom_Files', 'tet_1.atom'))

    lmp.close()

    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

    lmp_class.init_from_box(lmp)

    lmp_class.cg_min(lmp)

    pe_0 = lmp.get_thermo('pe')

    lmp.command('create_atoms 3 single %f %f %f units lattice' % (3.2500, 3.500, 2.000))

    lmp_class.cg_min(lmp)

    pe_1 = lmp.get_thermo('pe')

    print(pe_1 - pe_0)

    lmp.command('write_data %s' % os.path.join(output_folder, 'Data_Files', 'tet_2.data'))

    lmp.command('write_dump all custom %s id x y z'  % os.path.join(output_folder, 'Atom_Files', 'tet_2.atom'))

    natoms = lmp.get_natoms()

    lmp.close()

    init_path = os.path.join(output_folder, 'Data_Files', 'tet_1.data')

    final_path = os.path.join(output_folder, 'Atom_Files', 'tet_2.atom')

    with open(final_path, 'r') as file:
        lines = file.readlines()

    with open(final_path, 'w') as file:
        file.write(lines[3])
        file.writelines(lines[9:])

    neb_txt = gen_neb_inputfile(init_path, pottype, potfile, final_path, neb_image_folder, natoms)

    with open('%s.neb' % orient, 'w') as file:
        file.write(neb_txt)


if __name__ == '__main__':
    main()