import subprocess
import os
import glob
import numpy as np
from mpi4py import MPI
import sys 

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


def lammps_dump2data(inputfile):
    mpiprint ('''
Convert dump to restart file.


Max Boleininger 2020, mboleininger@gmail.com
    ''')


    exportfile = inputfile.split('/')[-1]
    exportfile = "%s.data" % exportfile.rpartition('.')[0]


    if os.path.exists(exportfile):
        announce ('File %s already exists.' % exportfile)
        return 0


    massdict = {'Fe': 55.845, 'W': 183.84}
    
    mass = massdict['W']

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


    return exportfile

comm = MPI.COMM_WORLD

proc_id = comm.Get_rank()

n_procs = comm.Get_size()

data_dir = '/home/ir-tiru1/rds/rds-ukaea-ap002-mOlK9qn0PlQ/CRAsimulations/Cascades/w_220_cascade'

files = glob.glob('%s/w_220_cascade.*.dump.gz' % data_dir)

dump_idx = sorted([int(file.split('.')[1]) for file in files])

chosen_idx = int(np.linspace(0, len(files) - 1, n_procs)[proc_id])

save_dir = '/home/ir-tiru1/Samanyu/Cascade_Simulations/w_220_cascade'

if not os.path.exists(save_dir):
        os.mkdir(save_dir)

file_gz = files[chosen_idx]

save_file = os.path.join(save_dir, os.path.basename(os.path.splitext(file_gz)[0]))

with open(save_file, 'wb') as out_file:
    subprocess.run(['gunzip', '-c', file_gz], stdout=out_file)

work_file = lammps_dump2data(save_file)

os.remove(save_file)

subprocess.run(['mpiexec','-n','1','./git_PolyCrystal_Analysis/build/bin/ws', '-f', work_file, '-N',
                 '0','220','220','220','0','220','220','220','0', '-a0', '3.1652'])

os.remove(work_file)