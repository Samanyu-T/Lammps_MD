import numpy as np
import sys, glob

def main():

    save_name = sys.argv[1]

    with open('log.lammps', 'r') as file:
        log = file.readlines()

    nprocs = int(log[1].split()[2])

    data = np.array([float(_str) for _str in log[-1].split()[9:]])

    data = data.reshape(len(data)//2, 2)

    data[:, 1] -= data[:,1].min()

    log_files = glob.glob('log.*.lammps')
    
    for i, _log_file in enumerate(log_files):
        with open(_log_file, 'r') as file:
            log = file.readlines()
            txt = log[-2].split(' ')
            data[i,0] = float(txt[-1])

    np.savetxt(save_name, data)

if __name__ == '__main__':
    main()