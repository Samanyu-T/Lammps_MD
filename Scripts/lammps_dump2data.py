# initialise
import os, sys, shutil, json, glob
import time
import numpy as np


sys.path.insert(0,'../..')


from lammps import lammps


# template to replace MPI functionality for single threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0


# try running in parallel, otherwise single thread
'''
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'
except:
'''
me = 0
nprocs = 1
comm = MPI_to_serial()
mode = 'serial'


def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0


def get_dump_frame(fpath):
    with open(fpath, 'r') as fp:
        fp.readline()
        frame = int(fp.readline())
    return frame


def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0


def main():
    mpiprint ('''
Convert dump to restart file.


Max Boleininger 2020, mboleininger@gmail.com
    ''')


    inputfile = sys.argv[1]
    exportfile = inputfile.split('/')[-1]
    exportfile = "%s.data" % exportfile.rpartition('.')[0]


    if os.path.exists(exportfile):
        announce ('File %s already exists.' % exportfile)
        return 0


    massdict = {'Fe': 55.845, 'W': 183.84}
    if len(sys.argv) == 3:
        if sys.argv[2] in massdict:
            mass = massdict[sys.argv[2]]
        else:
            mass = float(sys.argv[2])
    else:
        mass = 1.0
        announce ('warning: no mass specified, using 1.0 as placeholder.')
   
    # get list of unique atomic types
    iraw = []
    with open(inputfile) as ifile:
        for line in ifile:
            iraw += [line]
    iraw = iraw[9:]


    types = np.unique(np.array([_i.split()[1] for _i in iraw], dtype=np.int32))
    ntypes = len(types)


    # start lammps and import dump file
    lmp = lammps()


    lmp.command('# Lammps input file')
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
 
    #lmp.command('read_data %s' % _pf)
    lmp.command('region r_simbox block 0 1 0 1 0 1 units lattice')


    lmp.command('create_box %d r_simbox' % ntypes)
    for i in range(ntypes):
        lmp.command('mass %d %f' % (mass, i+1))


    iframe = get_dump_frame(inputfile)
    lmp.command('read_dump %s %d x y z add keep box yes' % (inputfile, iframe))
    lmp.command('write_data %s' % exportfile)


    lmp.close()


    return 0


if __name__ == "__main__":
    main()
    if mode == 'MPI':
        MPI.Finalize()