import numpy as np
import sys, os

def main():

    neb_script_file = sys.argv[1]

    save_name = neb_script_file.split('.')[-1]

    save_name += '.txt'

    with open('log.lammps', 'r') as file:
        log = file.readlines()

    nprocs = int(log[1].split()[2])

    data = np.array([float(_str) for _str in log[-1].split()[9:]])

    data = data.reshape(len(data)//2, 2)

    data[:, 1] -= data[:,1].min()

    np.savetxt(save_name, data)

if __name__ == '__main__':
    main()