import time
import numpy as np
import os
from lammps import lammps
import json
import sys
import glob
sys.path.append(os.path.join(os.getcwd(), 'git_folder','Classes'))
from sklearn.mixture import GaussianMixture
from Lammps_Classes import LammpsParentClass
from Handle_PotFiles import read_pot, write_pot
from scipy.optimize import minimize
import shutil
class ZBL():

    def __init__(self, Zi, Zj):
        
        self.Zi = Zi
        self.Zj = Zj

        e0 = 55.26349406e-4

        K = 1/(4*np.pi*e0)

        self.a = 0.46850/(self.Zi**0.23 + self.Zj**0.23)	

        self.amplitude = np.array([0.18175, 0.50986, 0.28022, 0.02817])
        self.exponent = np.array([3.19980, 0.94229, 0.40290, 0.20162])

        self.constant = K*Zi*Zj

    def eval_zbl(self, rij):

        if isinstance(rij, (int, float)):
            rij = np.array([rij])

        x = rij/self.a

        x = x[:, np.newaxis]

        phi = np.sum(self.amplitude * np.exp(-self.exponent * x), axis=1)
            
        return (self.constant/rij)*phi

    def eval_grad(self, rij):

        if isinstance(rij, (int, float)):
            rij = np.array([rij])

        x = rij/self.a

        x = x[:, np.newaxis]

        phi = np.sum(self.amplitude * np.exp(-self.exponent * x), axis=1)
        
        dphi = np.sum(-self.amplitude*self.exponent * np.exp(-self.exponent * x), axis=1)

        return (self.constant/rij)*(dphi/self.a - phi/rij)
    
    def eval_hess(self, rij):

        if isinstance(rij, (int, float)):
            rij = np.array([rij])
            
        x = rij/self.a

        x = x[:, np.newaxis]

        phi = np.sum(self.amplitude * np.exp(-self.exponent * x), axis=1)
        
        dphi = np.sum(-self.amplitude*self.exponent * np.exp(-self.exponent * x), axis=1)

        d2phi = np.sum(self.amplitude*self.exponent**2 * np.exp(-self.exponent * x), axis=1)

        return (self.constant/rij)*(d2phi/self.a**2 - 2*dphi/(self.a*rij) + 2*phi/rij**2)
    
def polyfit(x_arr, y_arr, dy_arr, d2y_arr):
    
    n_none = 0

    for lst in [y_arr, dy_arr, d2y_arr]:
        
        lst = lst.tolist()
        n_none += lst.count(None)
    
    dof = 3*len(x_arr) - n_none

    Phi = []
    Y   = []

    for i, x in enumerate(x_arr):

        y = y_arr[i]
        dy = dy_arr[i]
        d2y = d2y_arr[i]

        if y is not None:
            Phi.append(np.array([x**i for i in range(dof)]).T)
            Y.append(y)

        if dy is not None:
            Phi.append(np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T)
            Y.append(dy)

        if d2y is not None:
            Phi.append(np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T)
            Y.append(d2y)
        
    Phi = np.array(Phi)

    Y  = np.array(Y)

    return np.linalg.solve(Phi, Y)

def polyval(x, coef, func = True, grad = False, hess = False):

    dof = len(coef)

    Phi = np.array([])

    if func:
        Phi = np.array([x**i for i in range(dof)]).T

    elif grad:
        Phi = np.array([i*x**np.clip(i-1, a_min=0, a_max=None) for i in range(dof)]).T

    elif hess:
        Phi = np.array([i*(i-1)*x**np.clip(i-2, a_min=0, a_max=None) for i in range(dof)]).T

    if x.ndim == 1:
        return np.dot(Phi, coef)

    else:
        return np.dot(Phi, coef.reshape(-1,1)).flatten()

def splinefit(x_arr, y_arr, dy_arr, d2y_arr):
    
    coef_lst = []

    for i in range(len(x_arr) - 1):
        coef_lst.append(polyfit(x_arr[i:i+2], y_arr[i:i+2], dy_arr[i:i+2], d2y_arr[i:i+2]))
    
    return coef_lst

def splineval(x_arr, coef_lst, knots_pts, func = True, grad = False, hess = False):
    
    y = np.zeros(x_arr.shape)

    for i ,x in enumerate(x_arr): 
        idx = np.searchsorted(knots_pts, x) - 1

        if 0 <= idx < len(coef_lst):
            y[i] = polyval(x, coef_lst[idx], func, grad, hess).flatten()
        elif idx < 0:
            y[i] = coef_lst[0][0]
        else:
            y[i] = 0
    return y

def create_init_file(filepath):

    param = {
    "lattice_type": "bcc",
    "alattice": 3.144221,
    "C11": 3.201,
    "C12": 1.257,
    "C44": 1.020,
    "orientx": [
        1,
        1,
        0
    ],
    "orienty": [
        0,
        0,
        -1
    ],
    "orientz": [
        -1,
        1,
        0
    ],
    "size": 7,
    "surface": 20,
    "potfile": "git_folder/Potentials/test.eam.alloy",
    "conv": 100,
    "machine": "",
    "save_folder": "Monte_Carlo_HSurface"
}

    with open('init_param.json', "w") as json_file:
        json.dump(param, json_file, indent=4)

class Fit_EAM_Potential():

    def __init__(self, pot_lammps, n_knots, pot_params, potlines, comm, proc_id = 0, work_dir = ''):

        self.pot_lammps = pot_lammps

        self.work_dir = work_dir

        self.lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

        self.pot_folder = os.path.join(work_dir, 'Potentials')

        self.proc_id = proc_id

        self.keys  = ['He_F','He_p', 'W-He', 'He-He', 'H-He']

        self.lammps_param = {
                
        "lattice_type": "bcc",
        "alattice": 3.144237931,
        "C11": 3.201,
        "C12": 1.257,
        "C44": 1.020,
        "orientx": [
            1,
            1,
            0
        ],
        "orienty": [
            0,
            0,
            -1
        ],
        "orientz": [
            -1,
            1,
            0
        ],
        "size": 7,
        "surface": 20,
        "potfile": os.path.join(self.pot_folder, 'optim.%d.eam.alloy' % self.proc_id), #"git_folder/Potentials/test.eam.alloy",
        "conv": 1000,
        "machine": "",
        "save_folder": self.lammps_folder

        }

        self.pot_params = pot_params
        self.potlines = potlines
        self.bool_fit = {}

        for key in n_knots.keys():
            self.bool_fit[key] = bool(n_knots[key])

        self.n_knots = n_knots

        self.knot_pts = {}
        self.comm = comm

        self.knot_pts['He_F'] = np.linspace(0, self.pot_params['rho_c'], n_knots['He_F'])
        self.knot_pts['He_p'] = np.linspace(0, self.pot_params['rc'], n_knots['He_p'])
        self.knot_pts['W-He'] = np.linspace(0, self.pot_params['rc'], n_knots['W-He'])
        self.knot_pts['He-He'] = np.linspace(0, self.pot_params['rc'], n_knots['He-He'])
        self.knot_pts['H-He'] = np.linspace(0, self.pot_params['rc'], n_knots['H-He'])

        self.map = {}

        full_map_idx = [4*(n_knots['He_F'] - 2) + 1] + [4*(n_knots['He_p'] - 2) + 2] + [4*(n_knots['W-He'] - 2)] + [4*(n_knots['He-He'] - 2)] + [4*(n_knots['H-He'] - 2)]

        map_idx = []

        for idx, key in enumerate(self.bool_fit):
            if self.bool_fit[key]:
                map_idx.append(full_map_idx[idx])

        idx = 0
        iter = 0

        for key in self.keys:
            if self.bool_fit[key]:
                self.map[key] = slice(idx, idx + map_idx[iter])
                idx += map_idx[iter]
                iter += 1

        self.dof = idx

    def sample_to_array(self, sample_dict):
        
        sample_lst = []

        for key in self.keys:

            for val in sample_dict[key]:

                sample_lst.append(val)

        return np.array(sample_lst)
    
    def array_to_sample(self, sample_arr):
        
        sample_dict = {}

        for key in self.keys:
            sample_dict[key] = sample_arr[self.map[key]]

        return sample_dict
    
    def gen_rand(self):
            
        sample = np.zeros((self.dof,))

        ymax = 1
        dymax = 4
        d2ymax = 4

        if self.bool_fit['He_F']:
            
            x_bnds = np.linspace(0, self.pot_params['rho_c'], np.clip(self.n_knots['He_F'] - 1, a_min=2, a_max=np.inf).astype(int))

            sample[self.map['He_F']][0] = 20*(np.random.rand() - 0.5)

            for i in range(self.n_knots['He_F'] - 2):
                xmin = self.knot_pts['He_F'][i]
                xmax = self.knot_pts['He_F'][i + 1]

                sample[self.map['He_F']][4*i + 1] = (xmax - xmin)*np.random.rand() + xmin
                sample[self.map['He_F']][4*i + 2] = ymax*np.random.rand()
                sample[self.map['He_F']][4*i + 3] = dymax*(np.random.rand() - 0.5)
                sample[self.map['He_F']][4*i + 4] = d2ymax*(np.random.rand() - 0.5)


        ymax = 2
        dymax = 2
        d2ymax = 8

        x_bnds = np.linspace(0, self.pot_params['rc'], np.clip(self.n_knots['He_p'] - 1, a_min=2, a_max=np.inf).astype(int))

        if self.bool_fit['He_p']:

            # Randomly Generate Knot Values for Rho(r)
            sample[self.map['He_p']][0] = 5*np.random.rand()
            sample[self.map['He_p']][1] = -5*np.random.rand() 

            for i in range(self.n_knots['He_p'] - 2):

                xmin = self.knot_pts['He_p'][i]
                xmax = self.knot_pts['He_p'][i + 1]

                sample[self.map['He_p']][4*i + 2] = (xmax - xmin)*np.random.rand() + xmin
                sample[self.map['He_p']][4*i + 3] = ymax*(np.random.rand())
                sample[self.map['He_p']][4*i + 4] = -dymax*(np.random.rand())
                sample[self.map['He_p']][4*i + 5] = d2ymax*(np.random.rand() - 0.5)

        ymax = 4
        dymax = 10
        d2ymax = 20

        for key in ['W-He', 'He-He', 'H-He']:
            if self.bool_fit[key]:
                x_bnds = np.linspace(0, self.pot_params['rc'], np.clip(self.n_knots[key] - 1, a_min=2, a_max=np.inf).astype(int))

                # Randomly Generate Knot Values for Phi(r)
                for i in range(self.n_knots[key] - 2):
                    xmin = x_bnds[i]
                    xmax = x_bnds[i + 1]

                    sample[self.map[key]][4*i] = (xmax - xmin)*np.random.rand() + xmin
                    sample[self.map[key]][4*i + 1] = ymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][4*i + 2] = dymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][4*i + 3] = d2ymax*(np.random.rand() - 0.5)

        return sample
    
    def fit_sample(self, sample):

        coef_dict = {}

        if self.bool_fit['He_F']:
            
            x = np.copy(self.knot_pts['He_F'])

            y = np.zeros((self.n_knots['He_F'],))

            dy = np.full(y.shape, None, dtype=object)

            d2y = np.full(y.shape, None, dtype=object)

            # dy[0] = sample[self.map['He_F']][1]

            # d2y[0] = sample[self.map['He_F']][2]

            y[-1] = 0

            dy[-1] = 0

            d2y[-1] = 0

            for i in range(self.n_knots['He_F'] - 2):
                
                x[i + 1] = sample[self.map['He_F']][4*i + 1]

                y[i + 1] = sample[self.map['He_F']][4*i + 2] #- sample[0]*np.sqrt(x[i + 1])

                dy[i + 1] = sample[self.map['He_F']][4*i + 3] #- 0.5*sample[0]*x[i + 1]**(-0.5)

                d2y[i + 1] = sample[self.map['He_F']][4*i + 4] #+ 0.25*sample[0]*x[i + 1]**(-1.5)

            self.knot_pts['He_F'] = x

            coef_dict['He_F'] = splinefit(x, y, dy, d2y)

        if self.bool_fit['He_p']:

            x = np.copy(self.knot_pts['He_p'])

            y = np.zeros((self.n_knots['He_p'],))

            dy = np.full(y.shape, None, dtype=object)

            d2y = np.full(y.shape, None, dtype=object)

            y[0] = sample[self.map['He_p']][0]

            dy[0] = 0

            d2y[0] = sample[self.map['He_p']][1]

            y[-1] = 0

            dy[-1] = 0

            d2y[-1] = 0

            for i in range(self.n_knots['He_p'] - 2):
                
                x[i + 1] = sample[self.map['He_p']][4*i + 2]

                y[i + 1] = sample[self.map['He_p']][4*i + 3] 

                dy[i + 1] = sample[self.map['He_p']][4*i + 4]

                d2y[i + 1] = sample[self.map['He_p']][4*i + 5]

            self.knot_pts['He_p'] = x
            
            coef_dict['He_p'] = splinefit(x, y, dy, d2y)

        charge = [[74, 2],[2, 2],[1, 2]]

        for i, key in enumerate(['W-He', 'He-He', 'H-He']):

            if self.bool_fit[key]:

                zbl_class = ZBL(charge[i][0], charge[i][1])
                
                x = np.copy(self.knot_pts[key])

                y = np.zeros((len(x),))

                dy = np.zeros((len(x),))

                d2y = np.zeros((len(x),))

                for i in range(self.n_knots[key] - 2):

                    x[i + 1] = sample[self.map[key]][4*i]

                    y[i + 1] = sample[self.map[key]][4*i + 1] 

                    dy[i + 1] = sample[self.map[key]][4*i + 2]

                    d2y[i + 1] = sample[self.map[key]][4*i + 3]

                y[-1] = -zbl_class.eval_zbl(x[-1])[0]

                dy[-1] = -zbl_class.eval_grad(x[-1])[0]

                d2y[-1] = -zbl_class.eval_hess(x[-1])[0]

                self.knot_pts[key] = x

                coef_dict[key] = splinefit(x, y, dy, d2y)

        return coef_dict
    
    def sample_to_file(self, sample):

        coef_dict = self.fit_sample(sample)
        
        rho = np.linspace(0, self.pot_params['rho_c'], self.pot_params['Nrho'])

        r = np.linspace(0, self.pot_params['rc'], self.pot_params['Nr'])

        if self.bool_fit['He_F']:
            self.pot_lammps['He_F'] = splineval(rho, coef_dict['He_F'], self.knot_pts['He_F'], func = True, grad = False, hess = False)

        if self.bool_fit['He_p']:
            self.pot_lammps['He_p'] = splineval(r, coef_dict['He_p'], self.knot_pts['He_p'], func = True, grad = False, hess = False)

        charge = [[74, 2],[2, 2],[1, 2]]

        for i, key in enumerate(['W-He', 'He-He', 'H-He']):
            if self.bool_fit[key]:

                zbl_class = ZBL(charge[i][0], charge[i][1])
                
                zbl = zbl_class.eval_zbl(r[1:])

                poly = splineval(r[1:], coef_dict[key], self.knot_pts[key] , func = True, grad = False, hess = False)

                self.pot_lammps[key][1:] = r[1:]*(zbl + poly)

def sim_defect_set(optim_class:Fit_EAM_Potential):

    lmp_class = LammpsParentClass(optim_class.lammps_param, optim_class.comm, optim_class.proc_id)

    data = []
    
    files = glob.glob('%s/*.data' % optim_class.lammps_folder)
    
    for file in files:
        
        # V(1)H(3)He(6).(8).data
        filename = os.path.basename(file)

        vac = int(filename[1])

        h = int(filename[3])

        he = int(filename[6])

        image = int(filename[8])
        
        lmp_class.N_species = np.array([2*lmp_class.size**3 - vac, h, he])

        lmp = lammps(comm=optim_class.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(lmp_class.init_from_datafile(file)) 

        lmp_class.cg_min(lmp)

        ef = lmp_class.get_formation_energy(lmp)

        rvol = lmp_class.get_rvol(lmp)

        lmp.close()

        _data =  [vac, h, he, image, ef, rvol]
                
        data.append(_data)

    data = np.array(data)

    sort_idx = np.lexsort((data[:, 2], data[:, 1], data[:, 0], data[:, 3]))

    data = data[sort_idx]

    return data

def lst2matrix(lst):

    n_v = len(np.unique(lst[:,0]))
    n_h = len(np.unique(lst[:,1]))
    n_he = len(np.unique(lst[:,2]))
    n_image = len(np.unique(lst[:,3]))

    matrix = np.full((n_v, n_h, n_he, n_image, 2), np.inf)

    for row in lst:
        
        v, h, he, image, ef, rvol = row
        
        v = int(v)
        h = int(h)
        he = int(he)
        image = int(image)
        
        if not np.isnan(ef):
            matrix[v, h, he, image, 0] = ef

        if not np.isnan(rvol):
            matrix[v, h, he, image, 1] = rvol

    return matrix

def subtract_lst(x, y):
    res = []
    for _x, _y in zip(x, y):
        if not (np.isinf(_x) or np.isinf(_y)):
            res.append(_x - _y)
    return np.array(res)

def rel_abs_loss(y1, y2):
    loss = 0
    for i in range(min(len(y1), len(y2))):
        if y2[i] != 0:
            loss += np.abs(1 - y1[i]/y2[i])
    return loss

def loss_func(sample, data_ref, optim_class:Fit_EAM_Potential, diag=False):

    if diag:
        t1 = time.perf_counter()    

    # optim_class.sample_to_file(sample)

    # write_pot(optim_class.pot_lammps, optim_class.potlines, optim_class.lammps_param['potfile'])

    data_sample = sim_defect_set(optim_class)
 
    ref_mat = lst2matrix(data_ref)

    sample_mat = lst2matrix(data_sample)

    np.savetxt('test_sample.txt', data_sample, fmt='%.2f')

    loss = 0

    ''' 

    Loss from Helium Interstitial
    
    Image 0: Tet
    Image 1  Random Site - which optimizes to Tet
    Image 2: <110>
    Image 3: Oct
    Image 5: <111> 
    
    '''
    # Loss due to difference in Formation Energy
    loss += np.abs(1 - sample_mat[0, 0, 1, 0, 0]/ref_mat[0, 0, 1, 0, 0])

    loss += rel_abs_loss(sample_mat[0, 0, 1, 1:, 0] - sample_mat[0, 0, 1, 0, 0], ref_mat[0, 0, 1, 1:, 0] - ref_mat[0, 0, 1, 0, 0])

    # Loss due to difference in Relaxation Volume
    loss += np.abs(1 - sample_mat[0, 0, 1, 0, 1]/ref_mat[0, 0, 1, 0, 1])

    loss += np.abs(1 - sample_mat[0, 0, 1, 3, 1]/ref_mat[0, 0, 1, 3, 1])

    ''' Constraint '''

    constraint = not (np.arange(sample_mat.shape[3] - 1) == sample_mat[0, 0, 1, 1:, 0].argsort()).all()
    
    loss += 100*constraint  

    '''
    Loss from He-He Binding

    Interstital 
    Vacancy 
    Di-Vacancy

    '''
    for v in range(sample_mat.shape[0]):
            
        binding_sample = subtract_lst(np.min(sample_mat[v, 0, 1:, :, 0], axis = 1), np.min(sample_mat[v, 0, :-1, :, 0], axis = 1))
        
        binding_ref = subtract_lst(np.min(ref_mat[v, 0, 1:, :, 0], axis = 1), np.min(ref_mat[v, 0, :-1, :, 0], axis = 1))

        loss += rel_abs_loss(binding_sample, binding_ref)
    
        # print(v, 0 ,np.abs(subtract_lst(binding_sample, binding_ref) ) )

    '''
    Loss from H-He Binding

    Interstital 
    Vacancy 
    Di-Vacancy

    '''
    for v in range(sample_mat.shape[0]):

        for h in range(1, sample_mat.shape[1]):

            binding_sample = subtract_lst(np.min(sample_mat[v, h, :, :, 0], axis = 1), np.min(sample_mat[v, h-1, :, :, 0], axis = 1))
            
            binding_ref = subtract_lst(np.min(ref_mat[v, h, :, :, 0], axis = 1), np.min(ref_mat[v, h-1, :, :, 0], axis = 1))
            
            # print(v, h ,np.abs(subtract_lst(binding_sample, binding_ref) ) )

            loss += rel_abs_loss(binding_sample, binding_ref)

    ''' Loss from Relaxation Volumes '''

    for i in range(min(sample_mat.shape[0], ref_mat.shape[0])):
        for j in range(min(sample_mat.shape[1], ref_mat.shape[1])):
            for k in range(min(sample_mat.shape[2], ref_mat.shape[2])):
                for l in range(min(sample_mat.shape[3], ref_mat.shape[3])):

                    r_sample = sample_mat[i, j, k, l, 1]

                    r_ref = ref_mat[i, j, k, l, 1]

                    if not (np.isinf(r_ref) or np.isinf(r_sample) or r_ref == 0):
                        loss += abs(1 - (r_sample/r_ref))
    if diag:
        t2 = time.perf_counter()
        
        print(sample,loss, t2 - t1)

    return loss


def random_sampling(n_knots, comm, proc_id, max_time=3, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_data_new.txt')

    # Init Optimization Parameter
    t1 = time.perf_counter()

    sample = fitting_class.gen_rand()

    _ = loss_func(sample, data_ref, fitting_class)

    t2 = time.perf_counter()

    # if proc_id == 0:
    print('Average Time: %.2f s' % (t2 - t1))
    sys.stdout.flush()    
    
    lst_loss = []

    lst_samples = []

    t_init = time.perf_counter()

    idx = 0

    while True:

        sample = fitting_class.gen_rand()

        loss = loss_func(sample, data_ref, fitting_class)

        idx += 1

        lst_loss.append(loss)
        lst_samples.append(sample)
        
        t_end = time.perf_counter()
        
        if t_end - t_init > max_time:
            break

        if idx % 1000 == 0 and fitting_class.proc_id == 0:
            print(t_end - t_init)
            sys.stdout.flush()  

    lst_loss = np.array(lst_loss)
    lst_samples = np.array(lst_samples)

    idx = np.argsort(lst_loss)

    lst_loss = lst_loss[idx]
    lst_samples = lst_samples[idx]

    n = int( len(lst_loss) * 0.1 )

    np.savetxt(os.path.join(save_folder, 'Filtered_Samples_%d.txt' % proc_id), lst_samples[:n])
    np.savetxt(os.path.join(save_folder, 'Filtered_Loss_%d.txt' % proc_id), lst_loss[:n])



def gaussian_sampling(n_knots, comm, proc_id, mean, cov, max_time=3, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_data_new.txt')

    # Init Optimization Parameter
    t1 = time.perf_counter()

    sample = fitting_class.gen_rand()

    _ = loss_func(sample, data_ref, fitting_class)

    t2 = time.perf_counter()

    if proc_id == 0:
        print('Average Time: %.2f s' % (t2 - t1))
        sys.stdout.flush()    
    
    lst_loss = []

    lst_samples = []

    t_init = time.perf_counter()

    idx = 0

    while True:

        sample = np.random.multivariate_normal(mean=mean, cov=cov)

        loss = loss_func(sample, data_ref, fitting_class)

        idx += 1

        lst_loss.append(loss)
        lst_samples.append(sample)
        
        t_end = time.perf_counter()
        
        if t_end - t_init > max_time:
            break

        if idx % 1000 == 0 and fitting_class.proc_id == 0:
            print(t_end - t_init)
            sys.stdout.flush()  

    lst_loss = np.array(lst_loss)
    lst_samples = np.array(lst_samples)

    idx = np.argsort(lst_loss)

    lst_loss = lst_loss[idx]
    lst_samples = lst_samples[idx]

    n = int( len(lst_loss) * 0.1 )

    print(os.path.join(save_folder, 'Filtered_Samples_%d.txt' % proc_id))
    sys.stdout.flush()
    
    np.savetxt(os.path.join(save_folder, 'Filtered_Samples_%d.txt' % proc_id), lst_samples[:n])
    np.savetxt(os.path.join(save_folder, 'Filtered_Loss_%d.txt' % proc_id), lst_loss[:n])



def simplex(n_knots, comm, proc_id, x_init, maxiter = 100, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)


    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.alloy file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/test.eam.alloy')

    pot_params['rho_c'] = pot_params['Nrho']*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_data_new.txt')

    # Init Optimization Parameter
    t1 = time.perf_counter()

    _ = loss_func(x_init, data_ref, fitting_class)

    t2 = time.perf_counter()

    if proc_id == 0:
        print('Average Time: %.2f s' % (t2 - t1))
        sys.stdout.flush()    
    
    res = minimize(loss_func, x_init, args=(data_ref, fitting_class, True), method='Nelder-Mead', options={"maxiter":maxiter}, tol=1e-4)

    return res.x


def gmm(file_pattern, data_folder, iter):

    loss_lst = []  

    for file in glob.glob(os.path.join(file_pattern, 'Filtered_Loss_*.txt')):
        if os.path.getsize(file) > 0:
            loss_lst.append(np.loadtxt(file))
    
    loss = np.hstack([x for x in loss_lst])

    sample_lst = []  

    for file in glob.glob(os.path.join(file_pattern, 'Filtered_Samples_*.txt')):
        if os.path.getsize(file) > 0:
            sample_lst.append(np.loadtxt(file))
    
    samples = np.vstack([x for x in sample_lst])

    print(loss.mean(), loss.min())

    sort_idx = np.argsort(loss)
    loss = loss[sort_idx]
    samples = samples[sort_idx]

    thresh_idx = np.where(loss < 2*loss.min())[0]

    n = np.clip(10000, a_min = 0, a_max=len(thresh_idx)).astype(int)

    print(thresh_idx , n)
    data = samples[thresh_idx[:n]]

    cmp = 1
    gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=1e-6)
    gmm.fit(data)
    bic_val = gmm.bic(data)
    bic_val_prev = bic_val

    print(cmp, bic_val)
    sys.stdout.flush()  

    while True:

        cmp += 1
        bic_val_prev = bic_val
        gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=1e-6)
        gmm.fit(data)
        bic_val = gmm.bic(data)
        print(cmp, bic_val, bic_val_prev)

        if 1.01*bic_val > bic_val_prev:
            break

    print(cmp - 1)
    sys.stdout.flush()
      
    gmm = GaussianMixture(n_components=cmp - 1 , covariance_type='full')
    gmm.fit(data)

    print(data.shape)
    print(gmm.means_)

    gmm_folder = '%s/GMM_%d' % (data_folder, iter)

    if not os.path.exists(gmm_folder):
        os.mkdir(gmm_folder)

    np.savetxt(os.path.join(gmm_folder, 'Filtered_Loss.txt'),loss[thresh_idx])
    np.savetxt(os.path.join(gmm_folder, 'Filtered_Samples.txt'),samples[thresh_idx])

    np.save(os.path.join(gmm_folder, 'Cov.npy'),gmm.covariances_)
    np.save(os.path.join(gmm_folder, 'Mean.npy'),gmm.means_)

    return gmm.means_, gmm.covariances_