lmp_exec=~/lammps/build/lmp
mpiexec -np $(1) $(lmp_exec) -p $(1)x1 -in orient_100.neb
python read_neb_log.py orient_100.txt
mpiexec -np $(1) $(lmp_exec) -p $(1)x1 -in orient_110.neb
python read_neb_log.py orient_110.txt
mpiexec -np $(1) $(lmp_exec) -p $(1)x1 -in orient_111.neb
python read_neb_log.py orient_111.txt