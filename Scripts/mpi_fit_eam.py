from mpi4py import MPI
import sys
import time
import glob
import numpy as np
import os
import json
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
import EAM_Fitting
import shutil

def random_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time):

    ### START RANDOM SAMPLING ###
    rsamples_folder = ''

    if proc_id == 0:
        print('Start Random Sampling \n')
        sys.stdout.flush()  

    rsamples_folder = os.path.join(save_folder, 'Random_Samples') 

    if not os.path.exists(rsamples_folder) and proc_id == 0:
        os.mkdir(rsamples_folder)
        
    comm.Barrier()  
    
    t1 = time.perf_counter()
    EAM_Fitting.random_sampling(n_knots, comm_split, proc_id, max_time, work_dir, rsamples_folder)
    t2 = time.perf_counter()

    # Wait for the barrier to complete
    comm.barrier()
    
    if proc_id == 0:
        print('Random Sampling took %.2f s \n' % (t2 - t1))
        sys.stdout.flush()  
    
    comm.barrier()

    ### END RANDOM SAMPLING ###

    mean = None
    cov = None

    ### START CLUSTERING ALGORITHM ###
    if proc_id == 0:
        print('Start GMM Clustering \n')
        sys.stdout.flush()  

        t1 = time.perf_counter()

        mean, cov = EAM_Fitting.gmm(rsamples_folder, save_folder, 0)

        t2 = time.perf_counter()

        print('\n Clustering took %.2f s ' % (t2 - t1))
        sys.stdout.flush()  

    comm.barrier()

    mean = comm.bcast(mean, root = 0)

    cov = comm.bcast(cov, root=0)
    ## END CLUSTERING ALGORITHM ###
    
    return mean, cov

def gaussian_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time, g_iteration, N_iterations, mean, cov):

    for i in range(g_iteration, g_iteration + N_iterations):

        gsamples_folder = os.path.join(save_folder,'Gaussian_Samples_%d' % i)

        if proc_id == 0:
            print('Start Gaussian Sampling %dth iteration' % i)
            sys.stdout.flush()  

            if not os.path.exists(gsamples_folder):
                os.makedirs(gsamples_folder, exist_ok=True)

        comm.barrier()

        t1 = time.perf_counter()
        
        EAM_Fitting.gaussian_sampling(n_knots, comm_split, proc_id, mean[proc_id % mean.shape[0]], cov[proc_id % cov.shape[0]], max_time, work_dir, gsamples_folder)

        t2 = time.perf_counter()

        comm.barrier()

        if proc_id == 0:
            print('End Gaussian Sampling %dth iteration it took %.2f' % (i, t2- t1))
            sys.stdout.flush()  

            t1 = time.perf_counter()

            mean, cov = EAM_Fitting.gmm(gsamples_folder, save_folder, i + 1)

            t2 = time.perf_counter()

            print('\n Clustering took %.2f s ' % (t2 - t1))

            sys.stdout.flush()  

        comm.barrier()

        mean = comm.bcast(mean , root = 0)

        cov = comm.bcast(cov, root=0)
    
    return mean, cov


def extend_gmm(mean, cov, n):

    cov_base = np.array([4, 8, 16])

    mean_base = np.array([0 ,0, 0])

    cov_base = np.hstack([cov_base for i in range(n)]).flatten()

    mean_base = np.hstack([mean_base for i in range(n)]).flatten()

    cov_append = np.diag([cov_base])

    mean_append = mean_base

    cov_new = np.zeros((cov.shape[0], cov.shape[1] + cov_append.shape[0], cov.shape[1] + cov_append.shape[0]))

    mean_new = np.zeros((mean.shape[0], mean.shape[1] + mean_append.shape[0]))

    for i, _cov in enumerate(cov):

        cov_new[i] = np.block([[_cov, np.zeros((_cov.shape[0], cov_append.shape[0]))], 
                            [np.zeros((cov_append.shape[0], _cov.shape[0])), cov_append]])

    for i, _mean in enumerate(mean):

        mean_new[i] = np.hstack([_mean, mean_append])

    return mean_new, cov_new


def copy_files(w_he, he_he, h_he, work_dir, data_dir):
    
    data_files_folder = os.path.join(work_dir, 'Data_Files')

    if os.path.exists(data_files_folder):
        shutil.rmtree(data_files_folder)

    os.mkdir(data_files_folder)

    files_to_copy = []
    
    files_to_copy.extend(glob.glob('%s/V*H0He0.0.txt' % data_dir))

    if w_he:
        files_to_copy.extend(glob.glob('%s/V0H0He1.*.txt' % data_dir))
        files_to_copy.extend(glob.glob('%s/V*H0He1.0.txt' % data_dir))

    if he_he:
        files_to_copy.extend(glob.glob('%s/V*H0He*.*.txt' % data_dir))

    if h_he:
        files_to_copy.extend(glob.glob('%s/V*H*He*.*.txt' % data_dir))

    files_to_copy = list(set(files_to_copy))

    for file in files_to_copy:
        shutil.copy(file, data_files_folder)

def main(json_file):

    comm = MPI.COMM_WORLD

    proc_id = comm.Get_rank()

    n_procs = comm.Get_size()

    comm_split = comm.Split(proc_id, proc_id)

    with open(json_file, 'r') as file:
        param_dict = json.load(file)
    
    max_time = param_dict['max_time']
    work_dir = param_dict['work_dir']
    save_folder = param_dict['save_dir']
    data_dir = param_dict['data_dir']

    if (not os.path.exists(save_folder)) and proc_id == 0:
        os.mkdir(save_folder)

    if (not os.path.exists(work_dir)) and proc_id == 0:
        os.mkdir(work_dir)

    if (not os.path.exists(os.path.join(work_dir, 'Potentials'))) and proc_id == 0:
        os.mkdir(os.path.join(work_dir, 'Potentials'))

    comm.Barrier()

    n_knots = {}
    n_knots['He_F'] = 2
    n_knots['He_p'] = 0
    n_knots['W-He'] = 4
    n_knots['He-He'] = 0
    n_knots['H-He'] = 0

    if proc_id == 0:
        copy_files(True, False, False, work_dir, data_dir)

    comm.barrier()

    mean, cov = random_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time)

    ## START GAUSSIAN SAMPLING LOOP ###
    g_iteration = 0

    mean = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Mean.npy'))
    cov = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Cov.npy'))

    N_gaussian = 3
 
    mean, cov = gaussian_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time, g_iteration, N_gaussian, mean, cov)
    
    ## END GAUSSIAN SAMPLING LOOP ###

    ### OPTIMIZE FOR HE-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    g_iteration = 3

    N_gaussian = 4

    n_knots['He_F'] = 2
    n_knots['He_p'] = 0
    n_knots['W-He'] = 4
    n_knots['He-He'] = 3
    n_knots['H-He'] = 0

    if proc_id == 0:
        copy_files(True, True, False, work_dir, data_dir)
    comm.barrier()

    mean = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Mean.npy'))
    cov = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Cov.npy'))

    # Edit a new Covariance Matrix for the He-He potential
    if proc_id == 0:
        mean, cov = extend_gmm(mean, cov, n_knots['He-He'] - 2)
    comm.barrier()


    mean = comm.bcast(mean , root = 0)
    cov = comm.bcast(cov, root=0)


    ### END GAUSSIAN SAMPLING FOR HE-HE POTENTIAL ###
    mean, cov = gaussian_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time, g_iteration, N_gaussian, mean, cov)

    g_iteration = 7

    ### OPTIMIZE FOR H-HE POTENTIAL BY USING THE FINAL CLUSTER OF THE W-HE GMM AS A STARTING POINT ###
            
    n_knots['He_F'] = 2
    n_knots['He_p'] = 0
    n_knots['W-He'] = 4
    n_knots['He-He'] = 3
    n_knots['H-He'] = 4


    if proc_id == 0:
        copy_files(True, True, True, work_dir, data_dir)
    comm.barrier()

    # Edit a new Covariance Matrix for the H-He potential

    mean = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Mean.npy'))
    cov = np.load(os.path.join(save_folder, 'GMM_%d' % g_iteration, 'Cov.npy'))

    # Edit a new Covariance Matrix for the He-He potential
    if proc_id == 0:
        mean, cov = extend_gmm(mean, cov, n_knots['H-He'] - 2)
    comm.barrier()

    mean = comm.bcast(mean , root = 0)
    cov = comm.bcast(cov, root=0)


    ## BEGIN GAUSSIAN SAMPLING FOR H-HE POTENTIAL ###
    mean, cov = gaussian_sampling(comm, comm_split, proc_id, n_knots, save_folder, work_dir, max_time, g_iteration, N_gaussian, mean, cov)

    ### END GAUSSIAN SAMPLING FOR H-HE POTENTIAL ###

    g_iteration += N_gaussian

    if proc_id == 0:

        folders = glob.glob(os.path.join(save_folder, 'Gaussian_Samples_%d/Core_*' % (g_iteration - 1)))

        nprocs = len(folders)

        lst_samples = []
        lst_loss = []
        for folder in folders:
            lst_loss.append(np.loadtxt(os.path.join(folder, 'Filtered_Loss.txt')))
            lst_samples.append(np.loadtxt(os.path.join(folder, 'Filtered_Samples.txt')))

        loss = np.hstack(lst_loss).reshape(-1, 1)
        samples = np.vstack(lst_samples)

        N_simplex = 10

        for proc in range(nprocs):
            simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % proc)
            if not os.path.exists(simplex_folder):
                os.makedirs(simplex_folder, exist_ok=True)

        if nprocs >= len(loss):

            for i in range(len(loss)):
                simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % i)
                np.savetxt('%s/Simplex_Init.txt' % simplex_folder, samples[i])

            for i in range(len(loss), nprocs):
                simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % i)
                with open('%s/Simplex_Init.txt' % simplex_folder, 'w') as file:
                    file.write('')

        elif len(loss) > nprocs and N_simplex*nprocs > len(loss):
            part = len(loss) // nprocs
            idx = 0

            for proc in range(nprocs - 1):
                simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % proc)
                np.savetxt('%s/Simplex_Init.txt' % folders[proc], samples[idx: idx + part])
                idx += part

            simplex_folder = '%s/Simplex/Core_%d' % (save_folder, nprocs-1)
            np.savetxt('%s/Simplex_Init.txt' % folders[proc], samples[idx:])

        else:
            part = N_simplex
            sort_idx = np.argsort(loss.flatten())
            loss = loss[sort_idx]
            samples = samples[sort_idx]
            
            idx = 0
            for proc in range(nprocs):
                simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % proc)
                np.savetxt('%s/Simplex_Init.txt' % simplex_folder, samples[idx: idx + part])
                # print(samples[idx: idx + part])
                idx += part


    comm.Barrier()

    # simplex_folder = os.path.join(save_folder, 'Simplex/Core_%d' % me)
    # Simplex.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, simplex_folder=simplex_folder, work_dir=work_dir)
    # n_knots = [1, 1, 2]

    # bool_fit['He_F(rho)'] = bool(n_knots[0])
    # bool_fit['He_rho(r)'] = bool(n_knots[1])
    # bool_fit['W-He'] =   bool(n_knots[2])
    # bool_fit['H-He'] = True
    # bool_fit['He-He'] = True

    # gsamples_folder = os.path.join(save_folder, 'Gaussian_Samples_final')

    # t1 = time.perf_counter()

    # if me == 0:
    #     print('Start Gaussian Sampling last iteration')
    #     sys.stdout.flush()  

    #     if not os.path.exists(gsamples_folder):
    #         os.mkdir(gsamples_folder)

    # Gaussian_Sampling.optimize(n_knots=n_knots, bool_fit=bool_fit, proc=me, machine=machine, max_time=1.98*max_time,
    #                             work_dir=work_dir, sample_folder=gsamples_folder,
    #                             gmm_folder=os.path.join(save_folder,'GMM_final'))

    # t2 = time.perf_counter()

    # if me == 0:
    #     print(t2 - t1)


if __name__ == '__main__':
    main('fitting.json')