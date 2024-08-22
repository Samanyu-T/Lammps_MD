python git_folder/Scripts neb.py
mpiexec -np 5 ~/lammps/build/lmp -p 5x1 -in tet_tet.neb 
python git_folder/Scripts read_neb.py