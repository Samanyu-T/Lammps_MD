#!/bin/bash

num_procs=$1  # Assign the first argument to the number of processors
lmp_exec=/home/ir-tiru1/lammps/src/lmp_intel_cpu_intelmpi

# Run the first NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in orient100.neb
python read_neb_log.py orient_100.txt

# Run the second NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in orient110.neb
python read_neb_log.py orient_110.txt

# Run the third NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in orient111.neb
python read_neb_log.py orient_111.txt
