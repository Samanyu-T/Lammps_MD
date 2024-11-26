#!/bin/bash

num_procs=$1  # Assign the first argument to the number of processors
lmp_exec=/home/ir-tiru1/lammps/src/lmp_intel_cpu_intelmpi

# Run the first NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in sia_loop.neb
python git_folder/Scripts/read_neb_log.py sia_loop.txt

# Run the second NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in vac_loop.neb
python git_folder/Scripts/read_neb_log.py vac_loop.txt

# Run the third NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in edge_disloc.neb
python git_folder/Scripts/read_neb_log.py edge.txt

# Run the final NEB simulation
mpiexec -np $num_procs $lmp_exec -p ${num_procs}x1 -in screw_disloc.neb
python git_folder/Scripts/read_neb_log.py screw.txt
