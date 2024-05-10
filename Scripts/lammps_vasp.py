import numpy as np 
import os 
import glob 
def lammps_vasp(lammps_filepath, vasp_filepath):

    with open(lammps_filepath, 'r') as file:

        file.readline()
        file.readline()        
        file.readline()        
        file.readline()        
        file.readline()

        x_bounds = np.array( [float(string) for string in file.readline().split(' ')])
        y_bounds = np.array( [float(string) for string in file.readline().split(' ')])
        z_bounds = np.array( [float(string) for string in file.readline().split(' ')])

        bounds = np.row_stack([x_bounds, y_bounds, z_bounds])
    
    data = np.loadtxt(lammps_filepath, skiprows=9)

    type = data[:, 1]
    
    N_species = np.array([sum(type == species) for species in range(1, 4)])

    xyz = data[:, 2:]
    
    xyz[:, 0] -= bounds[0,0]
    xyz[:, 1] -= bounds[1,0]
    xyz[:, 2] -= bounds[2,0]
    
    pbc = bounds[:, 1] - bounds[:, 0]

    with open(vasp_filepath, 'w') as file:

        file.write('Based on %s \n' % lammps_filepath)

        file.write('1.0 \n')

        file.write('%.3f %.3f %.3f \n' % (pbc[0], 0, 0))

        file.write('%.3f %.3f %.3f \n' % (0, pbc[1], 0))

        file.write('%.3f %.3f %.3f \n' % (0, 0, pbc[2]))

        file.write('W H He \n')

        file.write('%d %d %d \n' % (N_species[0], N_species[1], N_species[2]) )

        file.write('Cartesian \n')

        for species in range(len(N_species)):

            xyz_species_idx = np.where(type == species + 1)[0]

            for idx in range(len(xyz_species_idx)):
                
                _xyz = xyz[xyz_species_idx[idx]]

                file.write('%.3f %.3f %.3f \n' % (_xyz[0], _xyz[1], _xyz[2]))


for filename in glob.glob('Lammps_Files_DFT_4x4/Atom_Files/*_mc.*.atom'):

    file_split = os.path.basename(filename).split('.')
    
    vasp_filename = os.path.join('VASP_files', '%s_snap_%s.poscar' % (file_split[0], file_split[1]))

    lammps_vasp(filename, vasp_filename)


    