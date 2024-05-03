from lammps import lammps
import os
import numpy as np
import json
from mpi4py import MPI
import ctypes
from scipy.spatial import KDTree
import ctypes 

class MPI_to_serial():

    def bcast(self, *args, **kwargs):

        return args[0]

    def barrier(self):

        return 0


class LammpsParentClass:
    """
    Parent class for the other Lammps methods, contains basic functions.
    """
    
    def __init__(self, init_dict, comm, proc_id = 0):
        """
        Initialize the Parent Class 

        Parameters:
        init_param_file (string): JSON file which initializes the Class - contains potential filepath, convergence params etc.
        """

        # Initialize the set of parameters from a JSON file

        self.comm = comm

        self.proc_id = proc_id

        for key in init_dict.keys():
            setattr(self, key, init_dict[key])

    def init_from_datafile(self, filepath):
        """
        Initialize a Lammps simulation from a .data file

        Parameters:
        filepath (string): Filepath of the Inital Configuration File.

        Returns:
        cmdlist (list): List of Lammps commands
        """

        cmdlist = []

        cmdlist.append('units metal')

        cmdlist.append('atom_style atomic')

        cmdlist.append('atom_modify map array sort 0 0.0')

        cmdlist.append('read_data %s' % filepath)

        cmdlist.append('pair_style eam/alloy')

        cmdlist.append('pair_coeff * * %s W H He' % self.potfile)

        cmdlist.append('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        cmdlist.append('run 0')

        return cmdlist
    
    def init_from_box(self):
        """
        Initialize a Lammps simulation by creating a box of atoms defined by the attributes of the Parent Class (JSON) file

        Returns:
        cmdlist (list): List of Lammps commands
        """

        cmdlist = []

        cmdlist.append('units metal')

        cmdlist.append('atom_style atomic')

        cmdlist.append('atom_modify map array sort 0 0.0')

        cmdlist.append('boundary p p p')

        cmdlist.append('lattice %s %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % 
                    (self.lattice_type, self.alattice,
                    self.orientx[0], self.orientx[1], self.orientx[2],
                    self.orienty[0], self.orienty[1], self.orienty[2], 
                    self.orientz[0], self.orientz[1], self.orientz[2]
                    ))

        cmdlist.append('region r_simbox block %f %f %f %f %f %f units lattice' % (
            -1e-9, self.size + 1e-9,
            -1e-9, self.size + 1e-9,
            -1e-9 - 0.5*self.surface, self.size + 1e-9 + 0.5*self.surface
        )
        )

        cmdlist.append('region r_atombox block %f %f %f %f %f %f units lattice' % (
            -1e-4, self.size + 1e-4,
            -1e-4, self.size + 1e-4,
            -1e-4, self.size + 1e-4
        )
        )

        cmdlist.append('create_box 3 r_simbox')
        
        cmdlist.append('create_atoms 1 region r_atombox')

        cmdlist.append('mass 1 183.84')

        cmdlist.append('mass 2 1.00784')

        cmdlist.append('mass 3 4.002602')

        cmdlist.append('pair_style eam/alloy' )

        cmdlist.append('pair_coeff * * %s W H He' % self.potfile)
        
        cmdlist.append('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')

        cmdlist.append('run 0')

        return cmdlist
    
    def cg_min(self, conv = None):
        """
        Set of commands that do a CG Minimization 

        Returns:
        cmdlist (list): List of Lammps commands
        """
        if conv is None:
            conv = self.conv

        cmdlist = []

        cmdlist.append('minimize 1e-9 1e-12 10 10')

        cmdlist.append('minimize 1e-12 1e-15 100 100')

        cmdlist.append('minimize 1e-13 1e-16 %d %d' % (conv, conv))

        return cmdlist
    
    def perfect_crystal(self):
        """
        Generate a perfect Tungsten crystal to use as reference for the rest of the point defects
        """
        
        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(self.init_from_box())

        # lmp.command('fix 3 all box/relax aniso 0.0')

        lmp.command('run 0')

        lmp.commands_list(self.cg_min(100000))

        lmp.command('write_data %s' % os.path.join(self.output_folder, 'Data_Files', 'V0H0He0.data'))
            
        self.stress_perfect = np.array([lmp.get_thermo('pxx'),
                                        lmp.get_thermo('pyy'),
                                        lmp.get_thermo('pzz'),
                                        lmp.get_thermo('pxy'),
                                        lmp.get_thermo('pxz'),
                                        lmp.get_thermo('pyz')
                                        ]) 
        
        # Update Lattice Constant after minimizing the box dimensions 
        self.alattice = lmp.get_thermo('xlat') / np.sqrt(np.dot(self.orientx, self.orientx))

        self.E_cohesive = np.array([-8.94964, -2.121, 0])
        
        self.N_species = np.array([lmp.get_natoms(), 0, 0])
        
        self.pe0 = lmp.get_thermo('pe')

        self.N_atoms0 = lmp.get_natoms()

        self.vol_perfect = lmp.get_thermo('vol')

        bounds = np.array([
            
            [lmp.get_thermo('xlo'), lmp.get_thermo('xhi')],
            [lmp.get_thermo('ylo'), lmp.get_thermo('yhi')],
            [lmp.get_thermo('zlo'), lmp.get_thermo('zhi')],

            ])

        self.pbc = (bounds[:,1].flatten() - bounds[:,0].flatten()) 

        print( (self.pe0 - self.E_cohesive[0]*self.N_atoms0)/(2*self.pbc[0]*self.pbc[1]) )
        self.offset = bounds[:,0].flatten()
        
        lmp.close()

    def run_MD(self, lmp, temp, timestep, N_steps):

        lmp.command('fix 1 all nve')

        rng_seed = np.random.randint(0, 10000)

        lmp.command('velocity all create %f %d mom yes rot no dist gaussian units box' % (temp, rng_seed))

        lmp.command('run 0')

        lmp.command('timestep %f' % timestep)

        lmp.command('run %d' % N_steps)

        lmp.command('velocity all zero linear')

    def get_formation_energy(self, lmp, N_species=None):

        pe = lmp.get_thermo('pe')

        if N_species is None:
            return pe - np.sum(self.E_cohesive*self.N_species)
        
        else:
            return pe - np.sum(self.E_cohesive*N_species)

    def trial_insert_atom(self, lmp, xyz_target, target_species, xyz_reset_c, return_xyz_optim = False):
        
        xyz_target -= self.offset
        
        xyz_target -= np.floor(xyz_target/self.pbc)*self.pbc

        xyz_target += self.offset

        lmp.command('create_atoms %d single %f %f %f units box' % 
                    (target_species, xyz_target[0], xyz_target[1], xyz_target[2])
                    )
                        
        lmp.commands_list(self.cg_min())
        
        lmp.command('write_dump all atom %s/test.atom' % self.output_folder)
        
        xyz_optim = None

        if return_xyz_optim:

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            xyz = xyz.reshape(len(xyz)//3, 3)

            xyz_optim = xyz[-1]

        n_atoms = lmp.get_natoms()

        N_species_copy = np.copy(self.N_species)

        N_species_copy[target_species - 1] += 1
        
        ef = self.get_formation_energy(lmp, N_species_copy)
            
        lmp.command('group del_atoms id %d' % (n_atoms))

        lmp.command('delete_atoms group del_atoms compress no')
        
        lmp.command('group del_atoms clear')

        lmp.scatter_atoms('x', 1, 3, xyz_reset_c)
        
        return ef, xyz_optim

    def get_kdtree(self, lmp, species_of_interest=None):
        """
        Finds the KDTree of the set of atoms in the simulation box

        Parameters:
        lmp (lammps attribute): Instance of lammps.lammps that is currently being used

        Returns:
        xyz (np.array): Positions of the atoms
        species (np.array): Species of each atom
        kdtree (scipy.spatial.KDTree): KDTree of all the atoms in the simulation
        pbc (np.array): The periodic box dimensions - offset such that the the periodic box starts at 0 and ends at 'pbc'
        offset (np.array): The offset applied to push the box to be on 0
        """
        xyz = np.array(lmp.gather_atoms('x', 1, 3))
        
        species = np.array(lmp.gather_atoms('type', 0, 1))
        
        self.N_species = np.array([sum(species == k) for k in range(1, 4)])

        xyz = xyz.reshape(len(xyz)//3, 3)

        xyz_kdtree = np.copy(xyz)

        bounds = np.array([
            
            [lmp.get_thermo('xlo'), lmp.get_thermo('xhi')],
            [lmp.get_thermo('ylo'), lmp.get_thermo('yhi')],
            [lmp.get_thermo('zlo'), lmp.get_thermo('zhi')],

            ])

        self.pbc = (bounds[:,1].flatten() - bounds[:,0].flatten()) 

        self.offset = bounds[:,0].flatten()

        xyz_kdtree -= self.offset

        kdtree = KDTree(xyz_kdtree, boxsize=self.pbc)

        return xyz, species, kdtree

    def get_rvol(self, lmp):
        """
        Finds the Relaxation Volume of the current Simulation - assuming an ideal perfect simulation has been built (GenerateStructure.perfect_crystal()

        Parameters:
        lmp (lammps attribute): Instance of lammps.lammps that is currently being used

        Returns:
        relaxation_vol (float): Relaxation Volume of the Simulation
        """

        # Find the Voigt Stress
        pxx = lmp.get_thermo('pxx')
        pyy = lmp.get_thermo('pyy')
        pzz = lmp.get_thermo('pzz')
        pxy = lmp.get_thermo('pxy')
        pxz = lmp.get_thermo('pxz')
        pyz = lmp.get_thermo('pyz')

        # vol = lmp.get_thermo('vol')

        stress_voigt = np.array([pxx, pyy, pzz, pxy, pxz, pyz]) - self.stress_perfect

        strain_tensor = self.find_strain(stress_voigt)
        
        relaxation_volume = 2*np.trace(strain_tensor)*self.vol_perfect/self.alattice**3

        return relaxation_volume
    
    def find_strain(self, stress_voigt):
        """
        Compute the strain from a given stress and the Elastic Tensor parameters in eV/A^2

        Parameters:
        stress_voigt (np array): Stress of the simulation box in Voigt Notation

        Returns:
        strain_tensor (np array): Strain in Tensor Notation
        """
        
        # Form the Elastic Tensor from the basic Moduli
        C = np.array( [
            [self.C11, self.C12, self.C12, 0, 0, 0],
            [self.C12, self.C11, self.C12, 0, 0, 0],
            [self.C12, self.C12, self.C11, 0, 0, 0],
            [0, 0, 0, self.C44, 0, 0],
            [0, 0, 0, 0, self.C44, 0],
            [0, 0, 0, 0, 0, self.C44]
        ])

        # Convert from eV/A^2 into MPa ??
        conversion = 1.602177e2

        C = conversion*C

        stress = stress_voigt*1e-4

        # Solve for the Strain
        strain_voigt = np.linalg.solve(C, stress)

        strain_tensor = np.array( [ 
            [strain_voigt[0], strain_voigt[3]/2, strain_voigt[4]/2],
            [strain_voigt[3]/2, strain_voigt[1], strain_voigt[5]/2],
            [strain_voigt[4]/2, strain_voigt[5]/2, strain_voigt[2]]
        ])

        return strain_tensor

    def add_defect(self, input_filepath, output_filepath, target_species, action, defect_centre):
        """
        Add a point defect to a given Simulation Box - aims to find the global minima but is not confirmed

        Parameters:
        input_filepath (string): Filepath of the Initial Configuration of which to add the defect to
        output_filepath (string): Filepath of the Output .data file
        target_species (integer): The species of the defect e.g a Tunsten defect would be 1, Hydrogen 2, Helium 3
        action (integer): Add (1) or remove (-1) an atom of the given species
        defect_centre (np array or None): The position of the defect in box units

        Returns:
        formation_energy (float): Formation Energy of the defect relative to a perfect crystal and molecular hydrogen and helium gas
        rvol (float): Relaxation Volume in lattice volume (a^3) relative a to a perfect crystal
        """
        
        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
        
        lmp.commands_list(self.init_from_datafile(input_filepath))
        
        lmp.command('run 0')

        # Get the KDTree of the simulation
        xyz, species, kdtree = self.get_kdtree(lmp)
        
        self.N_species = np.array([sum(species == k) for k in range(1, 4)])

        # If the defect centre is not specified simply use the centre of the simulation box - offset is added to bias the neighbours in a direction
        if defect_centre is None:

            defect_centre = ( self.offset + self.pbc/2 + np.array([-1e-2, -1e-2, 0]) )
        
        if defect_centre.ndim > 1:

            defect_centre = np.mean(defect_centre, axis=0)

        # Creates a vacancy by looping through each nearest-neighbors and selecting the energy minima
        if action == -1:
            
            # Find the closest species to the defect centre
            dist, nn = kdtree.query(defect_centre - self.offset, k=6)

            target_idxs = nn[species[nn] == target_species]
            
            pe_arr = np.zeros((len(target_idxs),))

            # Find the energy minima when each respective atom is removed 
            for i, idx in enumerate(target_idxs):
                    
                lmp.command('region sphere_remove_%d sphere %f %f %f 0.25' % 
                            (i,xyz[idx,0], xyz[idx,1], xyz[idx,2]))
                
                lmp.command('group del_atoms region sphere_remove_%d' % i)

                lmp.command('delete_atoms group del_atoms')
                
                lmp.command('group del_atoms clear')

                lmp.command('run 0')

                pe_arr[i] = lmp.get_thermo('pe')

                lmp.command('create_atoms %d single %f %f %f units box' % 
                            (target_species, xyz[idx,0], xyz[idx,1], xyz[idx,2])
                            )
            
            # Select the minima of NN and remove it
            optim_idx = target_idxs[np.round(pe_arr, 3).argmin()]            
            
            lmp.command('region sphere_remove sphere %f %f %f 0.25' % 
                        (xyz[optim_idx,0], xyz[optim_idx,1], xyz[optim_idx,2]))
            
            lmp.command('group del_atoms region sphere_remove')

            lmp.command('delete_atoms group del_atoms')
            
            lmp.command('group del_atoms clear')

            lmp.commands_list(self.cg_min())
            
        # Adds an interstitial to the system by trialling high symmettry points
        if action == 1:
            
            # Place the atom in optimal test site
            optim_site, min_pe  = self.random_config_minimizer(lmp, kdtree, xyz, species, defect_centre, target_species)
            
            lmp.command('create_atoms %d single %f %f %f units box' % 
                        (target_species, optim_site[0], optim_site[1], optim_site[2])
                        )
                        
            lmp.commands_list(self.cg_min())

        # Save Data Files and Atom Files for diagnosis and restart purposes
        lmp.command('write_data %s' % (output_filepath))
        
        dir_lst = output_filepath.split('/')
        
        delimiter = '/'

        atom_filepath = os.path.join(delimiter.join(dir_lst[:-2]), 'Atom_Files', '%s.atom' % dir_lst[-1].split('.')[0])

        lmp.command('write_dump all custom %s id type x y z' % (atom_filepath))

        pe_final = lmp.get_thermo('pe')

        species = np.array(lmp.gather_atoms('type', 0, 1))

        self.N_species = np.array([sum(species == k) for k in range(1, 4)])

        # Calculate the relaxation volume and formation energy
        rvol = self.get_rvol(lmp)

        formation_energy = self.get_formation_energy(lmp)
        
        return formation_energy, rvol
    
    def geometric_config_minimizer(self, lmp, kdtree, xyz, species, defect_centre, target_species):

        # Find a set of nearest-neighbours 
        n_neighbours = 9 + sum(species==2) + sum(species==3)
        
        dist, nn = kdtree.query(defect_centre - self.offset, k=n_neighbours)
                
        pe_lst = []
        
        sites_lst = []

        # Store the current state in memory to reset back to
        xyz_reset = lmp.numpy.extract_atom('x')
        
        xyz_c = xyz_reset.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # Geometric Test Sites based on any species
        for i in range(2, 6):
            
            test_site = np.mean(xyz[nn[:i]], axis = 0)   
            
            # print(dist)

            min_dist, closest_n = kdtree.query(test_site - self.offset, k = 1)
                        
            # Skip if too close to an atom
            if np.any(min_dist > 0.5):  
                sites_lst.append(test_site)

                lmp.command('create_atoms %d single %f %f %f units box' % 
                            (target_species, test_site[0], test_site[1], test_site[2])
                            )
                
                lmp.commands_list(self.cg_min())

                pe_lst.append(lmp.get_thermo('pe'))

                lmp.command('group del_atoms id %d' % (len(xyz) + 1))

                lmp.command('delete_atoms group del_atoms')
                
                lmp.command('group del_atoms clear')

                lmp.scatter_atoms('x', 1, 3, xyz_c)

        # Geometric Test Sites of positions of only the tungsten atoms
        slct_nn = nn[species[nn] == 1]

        for i in range(2, 6):
            
            test_site = np.mean(xyz[slct_nn[:i]], axis = 0)   

            min_dist, closest_n = kdtree.query(test_site - self.offset, k = 1)

            # Skip if too close to an atom
            if min_dist > 1:  
                sites_lst.append(test_site)

                lmp.command('create_atoms %d single %f %f %f units box' % 
                            (target_species, test_site[0], test_site[1], test_site[2])
                            )
                
                lmp.commands_list(self.cg_min())

                pe_lst.append(lmp.get_thermo('pe'))

                lmp.command('group del_atoms id %d' % (len(xyz) + 1))

                lmp.command('delete_atoms group del_atoms')
                
                lmp.command('group del_atoms clear')

                lmp.scatter_atoms('x', 1, 3, xyz_c)

        # Test the defect centre as a potential candidate
        if dist[0] > 0.5:

            lmp.command('create_atoms %d single %f %f %f units box' % 
                        (target_species, defect_centre[0], defect_centre[1], defect_centre[2])
                        )
            
            lmp.commands_list(self.cg_min())

            lmp.command('group del_atoms id %d' % (len(xyz) + 1))

            lmp.command('delete_atoms group del_atoms')
            
            lmp.command('group del_atoms clear')

            lmp.scatter_atoms('x', 1, 3, xyz_c)

            pe_lst.append(lmp.get_thermo('pe'))
            
            sites_lst.append(test_site)
        
        pe_lst = np.array(pe_lst)

        min_idx = pe_lst.argmin()

        return sites_lst[min_idx], pe_lst[min_idx]

    def random_config_minimizer(self, lmp, kdtree, xyz, species, defect_centre, target_species):
        
        if defect_centre.ndim == 1:
            defect_centre = defect_centre.reshape(1, -1)
        
        nn_list = []

        for pt in defect_centre:
            
            nn = kdtree.query_ball_point(pt - self.offset, self.alattice)

            nn_list.extend(nn)

        nn_list = np.unique(nn_list)

        min_radius_val = np.array([1.25, 0.75, 0.75])

        if len(nn_list) > 0:
            exclusion_centres = xyz[nn_list]

            exclusion_radii = np.array([min_radius_val[species[nn] - 1] for nn in nn_list])

        else:
            exclusion_centres = None

            exclusion_radii = None

        # Store the current state in memory to reset back to
        xyz_reset = lmp.numpy.extract_atom('x')
        
        xyz_reset_c = xyz_reset.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        n_to_accept = 0

        pe_lst = []

        sites_lst = []


        while True:

            rng = np.random.randint(low=0, high=defect_centre.shape[0])

            sample = self.alattice*(np.random.rand(3) - 0.5) + defect_centre[rng]

            if exclusion_centres is not None:
                delta = exclusion_centres - sample
                delta = delta - self.pbc * np.round(delta / self.pbc)
                dist_sample_centre = np.linalg.norm(delta, axis = 1)

            n_to_accept += 1

            min_dist, closest_n = kdtree.query(sample - self.offset, k = 1)
            
            if exclusion_radii is None:
                                
                n_to_accept = 0

                pe, xyz_min = self.trial_insert_atom(lmp, sample, target_species, xyz_reset_c, True)

                pe_lst.append(pe)

                sites_lst.append(sample)

                exclusion_centres = np.array([sample])
                
                radius = np.clip(np.linalg.norm(xyz_min - sample), a_min=0, a_max=2)

                exclusion_radii = np.array([radius])

            elif np.all(dist_sample_centre > exclusion_radii) and min_dist > 0.5:
                n_to_accept = 0

                pe, xyz_min = self.trial_insert_atom(lmp, sample, target_species, xyz_reset_c, True)

                pe_lst.append(pe)

                sites_lst.append(sample)

                exclusion_centres = np.vstack([exclusion_centres, sample])
                
                radius = np.clip(np.linalg.norm(xyz_min - sample), a_min=0, a_max=2)

                exclusion_radii = np.hstack([exclusion_radii, radius])

            if n_to_accept > 500 and len(pe_lst) > 1:

                pe_lst = np.array(pe_lst)

                min_idx = pe_lst.argmin()

                return sites_lst[min_idx], pe_lst[min_idx]

    def create_atoms_given_pos(self, input_filepath, output_filepath, target_species, target_xyz):
        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(self.init_from_datafile(input_filepath))
                
        lmp.command('run 0')

        for xyz, species in zip(target_xyz, target_species):

            lmp.command('create_atoms 1 ')

            lmp.command('create_atoms %d single %f %f %f units box' % 
                        (species, xyz[0], xyz[1], xyz[2])
                        )
            
        lmp.commands_list(self.cg_min())
        
        xyz = np.array(lmp.gather_atoms('x', 1, 3))

        pe = lmp.get_thermo('pe')

        rvol = self.get_rvol(lmp)

        # Save Data Files and Atom Files for diagnosis and restart purposes
        lmp.command('write_data %s' % (output_filepath))
        
        dir_lst = output_filepath.split('/')
        
        delimiter = '/'

        atom_filepath = os.path.join(delimiter.join(dir_lst[:-2]), 'Atom_Files', '%s.atom' % dir_lst[-1].split('.')[0])

        lmp.command('write_dump all custom %s id type x y z' % (atom_filepath))

        return pe, rvol, xyz[-len(target_species):]

    def fill_sites(self, input_filepath, region_of_interest, target_species, dist_between_sites, output_filename='init.data' ,int_energy = 0):

        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(self.init_from_datafile(input_filepath))
                
        lmp.command('run 0')

        xyz, species, kdtree = self.get_kdtree(lmp)
                
        slct_idx = np.where(np.all((xyz >= region_of_interest[0,:]) & (xyz <= region_of_interest[1,:]), axis=1))
        
        xyz_slct = xyz[slct_idx]

        kdtree_slct = KDTree(xyz_slct - self.offset, boxsize=self.pbc)

        avail_sites = np.arange(len(xyz_slct))
        
        while len(avail_sites) > 1:

            rng_idx = np.random.randint(low=0, high=len(avail_sites))

            idx = avail_sites[rng_idx]
            
            print(idx, lmp.get_natoms())
             
            for i in range(2):

                xyz_optim, pe_optim = self.random_config_minimizer(lmp, kdtree, xyz, species, xyz_slct[idx], target_species)
                                
                lmp.command('create_atoms %d single %f %f %f units box' % 
                        (target_species, xyz_optim[0], xyz_optim[1], xyz_optim[2])
                )

                lmp.commands_list(self.cg_min())

                self.N_species[target_species - 1] += 1
            
            nn = kdtree_slct.query_ball_point(xyz_slct[idx] - self.offset, r=dist_between_sites)
            
            avail_sites = avail_sites[np.bitwise_not(np.isin(avail_sites, nn))]
                        
        lmp.command('write_data %s' % os.path.join(self.output_folder, 'Data_Files', output_filename))

    def random_generate_atoms(self, input_filepath, region_of_interest, target_species, temp, output_filename):

        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(self.init_from_datafile(input_filepath))
                
        _, _, kdtree = self.get_kdtree(lmp)
        
        for i in range(len(target_species)):
            for j in range(target_species[i]):

                while True:
                    rng = np.random.rand(3)*(region_of_interest[1,:] - region_of_interest[0,:]) + region_of_interest[0,:]
                    
                    d, nn = kdtree.query(rng, k=1)

                    if d > 1:
                        lmp.command('create_atoms %d single %f %f %f units box' % 
                                    (i + 1, rng[0], rng[1], rng[2])
                                    )
                        break
        
        lmp.commands_list(self.cg_min())
        
        lmp.command('region vacuum block %f %f %f %f %f %f' % (
            self.offset[0], self.offset[0] + self.pbc[0],
            self.offset[1], self.offset[1] + self.pbc[1],
            self.offset[2], -5 ))
        

        for k in range(20):
            self.run_MD(lmp, temp=temp, timestep=1e-4, N_steps=1000)

            lmp.command('group del_atoms region vacuum')

            lmp.command('delete_atoms group del_atoms')

            lmp.command('group del_atoms clear')

        lmp.commands_list(self.cg_min())

        lmp.command('write_data %s' % os.path.join(self.output_folder, 'Data_Files', output_filename))

        print(os.path.join(self.output_folder, 'Data_Files', output_filename))

class Monte_Carlo_Methods(LammpsParentClass):



    def monte_carlo(self, input_filepath, species_of_interest, N_steps, p_events_dict, 
                    temp = 300, potential = 0, max_displacement = np.ones((3,)),
                    region_of_interest= None, save_xyz=False, diag = False):
        """
        Applies a Monte Carlo Algorithm to the given Simulation Box - when everything is active it will result in the semi-grand canonical ensemble

        Parameters:
        input_filepath (string): Location of Lammps Input Data File
        species_of_interest (list of int): List of atom species of interest
        N_steps (int): Number of accepted data-points
        p_events_dict (dict): Contains the Probability of Monte Carlo Events: [p_displace, p_exchange, p_delete, p_create] - should sum up to one
        temp (float): Temperature of MC simulation
        potential (float): Applied Chemical Potential
        max_displacement (ndarray): The maximum displacement in each direction [x_max, y_max, z_max] - randomly sample displacement from (-x_max, x_max)
        region_of_interest (ndarray): Only applicable when creating atoms (for now)
        save_xyz (bool): Boolean for saving the positions of the structures
        diag (bool): Boolean for printing diagnositics

        Returns:
        pe_accept_ar (np.array): The potential energies of the accepted configurations
        rvol_accept_arr (np.array): The xyz co-ordinates of the accepted configurations
        xyz_accept_lst (list[np.array]): The xyz co-ordinates of the accepted configurations
        acceptance_ratio (float): Acceptance ratio of the MCMC simulatoin
        """

        # init params and data arrays
        n_iterations = 0

        kb = 8.6173303e-5

        n_accept = 0

        self.beta = 1/(kb*temp)

        pe_accept_arr = np.zeros((N_steps + 1, ))
        
        rvol_accept_arr = np.zeros((N_steps + 1, ))
        
        n_species_accept_arr = np.zeros((N_steps + 1, 3))

        xyz_accept_lst = []

        if not isinstance(species_of_interest, list):
            species_of_interest = [species_of_interest]

        if isinstance(potential, float):
            potential = potential*np.ones((3,))

        key = {'displace':0, 'exchange':1, 'delete':2, 'create':3}

        p_events = np.zeros((len(p_events_dict.keys())))

        for key_loop in p_events_dict.keys():
            p_events[key[key_loop]] = p_events_dict[key_loop]

        p_cumulative = np.cumsum(p_events)

        self.potential = potential

        # init lammps simulation
        lmp = lammps(name = self.machine, comm=self.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(self.init_from_datafile(input_filepath))
        
        lmp.command('run 0')

        species = np.array(lmp.gather_atoms('type', 0, 1))
        
        N_species = np.array([sum(species == k) for k in range(1, 4)])
        
        # store the current state
        self.pe_current = self.get_formation_energy(lmp, N_species)

        rvol = self.get_rvol(lmp)

        n_species_accept_arr[0] = N_species

        rvol_accept_arr[0] = rvol

        pe_accept_arr[0] = self.pe_current

        p_events_copy = np.copy(p_events)

        if save_xyz:
            xyz_accept_lst.append(np.array(lmp.gather_atoms('x',1,3)))

        if region_of_interest is None:

            region_of_interest = np.row_stack((self.offset, self.pbc + self.offset))

        while n_accept < N_steps and n_iterations < N_steps*10:
            
            # Increase the loop counter
            n_iterations += 1

            # Get the KDTree of the simulation
            species = np.array(lmp.gather_atoms('type', 0, 1))

            self.N_species = np.array([sum(species == k) for k in range(1, 4)])

            condition = np.logical_or.reduce([species == val for val in species_of_interest])

            slct_atom_idx = np.where(condition)[0]
            
            # print(self.pe_current, self.N_species)

            # if there are no atoms that are selected set probability of exchange, delete and displacement actions to 0 and probability of creation to 1
            if len(slct_atom_idx) == 0:
                
                if p_events[key['create']] == 0:

                    return pe_accept_arr[:n_accept + 1], rvol_accept_arr[:n_accept + 1], xyz_accept_lst, n_accept/n_iterations
                
                else:
                    p_events_copy = np.zeros((4,))

                    p_events_copy[key['create']] = 1

                    p_cumulative = np.cumsum(p_events_copy)

            # if there are only a single type of species in the set of particles - then exchange cannot be possible
            elif np.bitwise_or.reduce([N_species[target - 1] == 0 for target in species_of_interest]):
                
                p_norm = np.sum([ p_events[key[key_loop]] for key_loop in key.keys() if key_loop != 'exchange' ]) 

                p_coef = (1 + p_events[key['exchange']]/p_norm)

                for i in range(len(key)):

                    if i == key['exchange']:
                        p_events_copy[key['exchange']] = 0

                    else:
                        p_events_copy[i] = p_events[i] * p_coef

                p_cumulative = np.cumsum(p_events_copy)

            
            # reset the probability of events
            else:
                p_cumulative = np.cumsum(p_events)


            # store the current state in memory to reset back to
            xyz_reset = lmp.numpy.extract_atom('x')
            
            self.xyz_reset_c = xyz_reset.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            # randomly generate probabiltiy of event and shuffle the atomic indexes            
            np.random.shuffle(slct_atom_idx)

            event = np.searchsorted(p_cumulative, np.random.rand(), side='left')

            if event == key['exchange']:

                idx_h = np.where(species[slct_atom_idx] == 2)[0][0]

                idx_he = np.where(species[slct_atom_idx] == 3)[0][0]

                acceptance, pe_test, N_species_test = self.exchange_atoms(lmp, slct_atom_idx[idx_h], slct_atom_idx[idx_he])

            elif event == key['delete']:

                idx_delete = slct_atom_idx[0]

                species_delete = species[idx_delete]

                xyz_delete = xyz_reset[idx_delete]

                acceptance, pe_test, N_species_test = self.delete_atoms(lmp, idx_delete, species_delete, xyz_delete)

            elif event == key['displace']:
                
                idx_displace = slct_atom_idx[0]

                acceptance, pe_test, N_species_test = self.displace_atoms(lmp, idx_displace, max_displacement)

            elif event == key['create']:
                            
                species_create = species_of_interest[np.random.randint(0,len(species_of_interest))]

                xyz_create = region_of_interest[1,:]*np.random.rand(3) + region_of_interest[0,:]

                acceptance, pe_test, N_species_test = self.create_atoms(lmp, species_create, xyz_create)

            # print(event)
            if acceptance:

                n_accept += 1

                self.pe_current = pe_test

                pe_accept_arr[n_accept] = pe_test

                n_species_accept_arr[0] = N_species_test

                rvol = self.get_rvol(lmp)   

                rvol_accept_arr[n_accept] = rvol

                if save_xyz:
                    xyz_accept_lst.append(np.array(lmp.gather_atoms('x',1,3)))
                                
                if n_accept % 2 == 0 and diag == True:

                    print(n_accept, self.pe_current - pe_accept_arr[-1])

                    filename = '%s.%d.atom' % (os.path.basename(input_filepath).split('.')[0], n_accept)

                    lmp.command('write_dump all custom %s id type x y z' % os.path.join(self.output_folder, 'Atom_Files', filename))


        filename_data = '%s.final.data' % (os.path.basename(input_filepath).split('.')[0])
        filename_atom = '%s.final.atom' % (os.path.basename(input_filepath).split('.')[0])

        lmp.command('write_data %s' % os.path.join(self.output_folder,'Data_Files', filename_data))

        lmp.command('write_dump all custom %s id type x y z' % os.path.join(self.output_folder,'Atom_Files', filename_atom))
        
        lmp.close()

        return pe_accept_arr, rvol_accept_arr, n_species_accept_arr, xyz_accept_lst, n_accept/n_iterations
    
    def mc_acceptance_criterion(self, lmp, delta_N, N_species):

        lmp.commands_list(self.cg_min())

        pe_test = self.get_formation_energy(lmp, N_species)

        pe_delta = pe_test - self.pe_current + np.sum(self.potential*delta_N)
        
        exponent = np.clip(self.beta * pe_delta, a_min=0, a_max=10)

        acceptance = np.exp(-exponent) 
        
        p_accept = np.random.rand()
        
        # print(self.N_species, pe_delta)

        return p_accept < acceptance, pe_test


    def displace_atoms(self, lmp, idx, displacement_coef):

        lmp.command('group displace id %d' % idx)

        disp = displacement_coef*self.alattice*(np.random.rand(3) - 0.5)
        
        lmp.command('displace_atoms displace move %f %f %f' % (disp[0], disp[1], disp[2]) )

        lmp.command('group displace clear')

        delta_N = np.zeros((3,))

        acceptance, pe_test = self.mc_acceptance_criterion(lmp, delta_N, self.N_species)

        if not acceptance:
            
            lmp.scatter_atoms('x', 1, 3, self.xyz_reset_c)

            lmp.command('run 0')


        return acceptance, pe_test, self.N_species
    
    def delete_atoms(self, lmp, idx, species_delete, xyz_delete):

        lmp.command('group del_atoms id %d' % (idx + 1))

        lmp.command('delete_atoms group del_atoms')
        
        lmp.command('group del_atoms clear')

        delta_N = np.zeros((3,))

        delta_N[species_delete - 1] = -1
        
        N_species = self.N_species + delta_N

        acceptance, pe_test = self.mc_acceptance_criterion(lmp, delta_N, N_species)

        if not acceptance:

            lmp.command('create_atoms %d single %f %f %f units box' % 
                        (species_delete, xyz_delete[0], xyz_delete[1], xyz_delete[2])
                        )
            
            lmp.scatter_atoms('x', 1, 3, self.xyz_reset_c)

            lmp.command('run 0')
        
        lmp.command('reset_atoms id')

        return acceptance, pe_test, N_species

    def create_atoms(self, lmp, species_create, xyz_create):
        
        lmp.command('run 0')

        xyz, species, kdtree = self.get_kdtree(lmp)
                
        xyz_optim, pe_optim = self.random_config_minimizer(lmp, kdtree, xyz, species, xyz_create, species_create)

        lmp.command('create_atoms %d single %f %f %f units box' % 
                    (species_create, xyz_optim[0], xyz_optim[1], xyz_optim[2])
                    )
        
        
        # lmp.command('write_dump all custom %s id type x y z' % os.path.join(self.output_folder,'Atom_Files', 'test.atom'))
        
        min_dist, closest_n = kdtree.query(xyz_optim - self.offset, k = 1)

        # print(min_dist, lmp.get_natoms())

        delta_N = np.zeros((3,))

        delta_N[species_create - 1] += 1
        
        N_species = self.N_species + delta_N

        acceptance, pe_test = self.mc_acceptance_criterion(lmp, delta_N, N_species)

        if not acceptance:

            lmp.command('group del_atoms id %d' % (lmp.get_natoms()))
 
            lmp.command('delete_atoms group del_atoms')
            
            lmp.command('group del_atoms clear')

            lmp.scatter_atoms('x', 1, 3, self.xyz_reset_c)

            lmp.command('run 0')
                    
        lmp.command('reset_atoms id')

        return acceptance, pe_test, N_species


    def exchange_atoms(self, lmp, idx_h, idx_he):

        lmp.command('set atom %d type %d' % (idx_h + 1, 3) )
        
        lmp.command('set atom %d type %d' % (idx_he + 1, 2) )
        
        delta_N = np.zeros((3,))

        acceptance, pe_test = self.mc_acceptance_criterion(lmp, delta_N, self.N_species)

        if not acceptance:
                
            lmp.command('set atom %d type %d' % (idx_h + 1, 2) )
            
            lmp.command('set atom %d type %d' % (idx_he + 1, 3) )
                
            lmp.scatter_atoms('x', 1, 3, self.xyz_reset_c)

            lmp.command('run 0')

        return acceptance, pe_test, self.N_species

    def hybridx_mcmd(self, init_config_file_path, temp = 800):

        lmp = lammps()#name = self.machine, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.command('# Lammps input file')

        lmp.command('units metal')

        lmp.command('atom_style atomic')

        lmp.command('atom_modify map array sort 0 0.0')

        lmp.command('read_data %s' % init_config_file_path)

        lmp.command('pair_style eam/alloy')

        lmp.command('pair_coeff * * %s W H He' % self.potfile)
        
        lmp.command('compute pe_atom all pe/atom')

        lmp.command('thermo 100')

        lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
        
        kb = 8.6173e-5

        beta = 1/(kb*temp)

        n_accept = 0
        
        counter = 0

        N_atoms = lmp.get_natoms()

        species = np.array(lmp.gather_atoms('type', 0, 1))

        light_idx = np.arange(N_atoms)[species != 1]

        max_move = 1

        lmp.command('run 0')

        pe_curr = lmp.get_thermo('pe')

        pe_test = 0

        accepted_pe = [pe_curr]

        all_pe = [pe_curr]
        
        rng_seed = np.random.randint(1,10000)
        
        lmp.command('fix thermostat all nvt temp %f %f $(100.0*dt)' % (temp, temp))
        
        lmp.command('velocity all create %f %d mom yes rot no dist gaussian units box' % (2*temp, rng_seed))

        lmp.command('run 0')

        lmp.command('timestep 1e-4')

        lmp.command('run 50000')

        print(lmp.get_thermo('temp'))

        lmp.command('velocity all zero linear')

        while n_accept < 100:
            lmp.command('write_dump all atom ../MCMC_Dump/Nh_%d_image_%d.atom' % (len(light_idx), counter))

            counter += 1

            xyz = np.array(lmp.gather_atoms('x', 1, 3))

            force = np.array(lmp.gather_atoms('f', 1, 3))

            xyz = xyz.reshape(len(xyz)//3 , 3)

            force = force.reshape(len(force)//3 , 3)

            np.random.shuffle(light_idx)

            slct_h = np.clip(len(light_idx), a_min=0, a_max=max_move)

            kick_idx = np.copy(light_idx[:slct_h])
            
            random_kick = (np.random.rand(len(kick_idx),3) - 0.5)
            
            force_unit_vector = force[kick_idx]/np.clip(np.linalg.norm(force[kick_idx], axis = 1, keepdims=True), a_min = 1e-6, a_max=np.inf)

            kick_unit_vector = -0*force_unit_vector + 1*random_kick

            kick_magnitude = 2*self.alattice*np.random.rand(len(kick_idx))
            
            kick_vector = kick_magnitude[:,np.newaxis]*kick_unit_vector

            kick_vector[:,-1] = np.clip(kick_unit_vector[:,-1], a_min=-10, a_max=0.5)

            xyz[kick_idx] += kick_vector
            
            xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            lmp.scatter_atoms('x', 1, 3, xyz_c)
            
            lmp.commands_list(self.cg_min())

            pe_test = lmp.get_thermo('pe')

            acceptance = np.min([1, np.exp(-beta*(pe_test - pe_curr))])

            rng = np.random.rand()

            if rng <= acceptance: 
                
                pe_curr = pe_test

                accepted_pe.append(pe_curr)

                n_accept += 1
                
                print('accept: ',pe_test - accepted_pe[0])

                lmp.command('write_dump all atom ../MCMC_Dump/Nh_%d_image_%d.atom' % (len(light_idx), n_accept))
                
            else:
                
                xyz[kick_idx] -= kick_vector

                xyz_c = xyz.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

                lmp.scatter_atoms('x', 1, 3, xyz_c)
                
                print('reject: ',pe_test - accepted_pe[-1])

