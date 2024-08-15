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
from Handle_PotFiles_He import read_pot, write_pot
from scipy.optimize import minimize, basinhopping, differential_evolution
import shutil
import copy
from scipy.integrate import simpson
from scipy.interpolate import interp1d

def eval_virial(phi, T_arr, r):

    virial_coef = np.zeros(T_arr.shape)
    
    kb = 8.6173303e-5
    
    conv = 6.02214e-1

    for i, T in enumerate(T_arr):

        beta = 1 / (kb * T)

        exponent = np.clip(beta * phi, a_min = -10, a_max = 10)

        y = ( 1 - np.exp(- exponent) ) * r**2

        virial_coef[i] = 2* np.pi * conv * simpson(y,x=r)

    return virial_coef

def gauss(x, A, sigma):
    A = abs(A)
    sigma = abs(sigma)
    return A * np.exp(-0.5*(x/sigma)**2)

def dgauss(x, A, sigma):
    A = abs(A)
    sigma = abs(sigma)
    return - (A * x / sigma ** 2) * np.exp(-0.5*(x/sigma)**2)

def d2gauss(x, A, sigma):
    A = abs(A)
    sigma = abs(sigma)
    return (A / sigma ** 4) * (x**2 - sigma**2) * np.exp(-0.5*(x/sigma)**2)

def f_1s(x, Z, c):
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c*Z**4/(k*np.pi*a0**3)
    b = (2*Z/a0)
    return A * np.exp(-b * x)

def df_1s(x, Z, c):
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c*Z**4/(k*np.pi*a0**3)
    b = (2*Z/a0)
    return -b * A * np.exp(-b * x)

def d2f_1s(x, Z, c):
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c*Z**4/(k*np.pi*a0**3)
    b = (2*Z/a0)
    return b**2 * A * np.exp(-b * x)

def f_2p(x, Z, c):
    c = 1 - c
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c * (Z/(96*np.pi*k))* (Z/a0)**5
    b = Z/a0
    return A * x**2 * np.exp(-b * x)


def df_2p(x, Z, c):
    c = 1 - c
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c * (Z/(96*np.pi*k))* (Z/a0)**5
    b = Z/a0
    return A * (2*x - b*x**2) * np.exp(-b * x)


def d2f_2p(x, Z, c):
    c = 1 - c
    Z = abs(Z)
    a0 = 0.529
    k = 0.1366
    A = c * (Z/(96*np.pi*k))* (Z/a0)**5
    b = Z/a0
    return A * (2 - 4*b*x + b**2 * x**2) * np.exp(-b * x)

def exp(x, A, b):
    A = abs(A)
    b = abs(b)
    return A * np.exp(-b * x)

def dexp(x, A, b):
    A = abs(A)
    b = abs(b)
    return -b * A * np.exp(-b * x)

def d2exp(x, A, b):
    A = abs(A)
    b = abs(b)
    return b**2 * A * np.exp(-b * x)


def nexp(x, A, b, c):
    A = abs(A)
    b = abs(b)
    c = abs(c)
    return A * np.exp(-b * x) * (1 - c*x)

def dnexp(x, A, b, c):
    A = abs(A)
    b = abs(b)
    c = abs(c)
    return A * np.exp(-b * x) * (-c -b + b*c*x)

def d2nexp(x, A, b, c):
    A = abs(A)
    b = abs(b)
    c = abs(c)
    return A * np.exp(-b * x) * (b**2 + 2*b*c - c*b**2*x)

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
    "size": 4,
    "surface": 0,
    "potfile": "git_folder/Potentials/init.eam.he",
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

        self.keys  = ['He F','H-He p', 'He-W p', 'He-H p', 'He-He p', 'W-He', 'He-He', 'H-He', 'W-He p']

        self.lammps_param = {
                
        "lattice_type": "bcc",
        "alattice": 3.14421,
        "C11": 3.201,
        "C12": 1.257,
        "C44": 1.020,
        "orientx": [
            1,
            0,
            0
        ],
        "orienty": [
            0,
            1,
            0
        ],
        "orientz": [
            0,
            0,
            1
        ],
        "size": 4,
        "surface": 0,
        "potfile": os.path.join(self.pot_folder, 'optim.%d.eam.he' % self.proc_id), #"git_folder/Potentials/init.eam.he"
        "pottype":"he",
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

        self.knot_pts['He F'] = np.linspace(self.pot_params['rhomin'], self.pot_params['rho_c'], n_knots['He F'])

        if n_knots['He F'] > 2:
            self.knot_pts['He F'][1] = 0.3

        self.knot_pts['H-He p'] = np.linspace(0, self.pot_params['rc'], n_knots['H-He p'])
        self.knot_pts['He-W p'] = np.linspace(0, self.pot_params['rc'], n_knots['He-W p'])
        self.knot_pts['He-H p'] = np.linspace(0, self.pot_params['rc'], n_knots['He-H p'])
        self.knot_pts['He-He p'] = np.linspace(0, self.pot_params['rc'], n_knots['He-He p'])

        self.knot_pts['W-He'] = np.linspace(0, self.pot_params['rc'], n_knots['W-He'])
        if n_knots['W-He'] == 4:
            self.knot_pts['W-He'][1:3] = np.array([1.7581, 2.7236])
        self.knot_pts['He-He'] = np.linspace(0, self.pot_params['rc'], n_knots['He-He'])
        self.knot_pts['H-He'] = np.linspace(0, self.pot_params['rc'], n_knots['H-He'])

        self.knot_pts['W-He p'] = np.array([0, 3.27332980, 4.64995591])

        self.map = {}

        full_map_idx = [3*(n_knots['He F'] - 2) + 2] + [3*(n_knots['H-He p'] - 2) + 3] + \
                       [3*(n_knots['He-W p'] - 2) + 3] + [3*(n_knots['He-H p'] - 2) + 3] + [3*(n_knots['He-He p'] - 2) + 3] + \
                       [3*(n_knots['W-He'] - 2)] + [3*(n_knots['He-He'] - 2)] + [3*(n_knots['H-He'] - 2)] + \
                       [3*(n_knots['W-He p'] - 1)]        
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

        ymax = 2
        dymax = 4
        d2ymax = 4
        
        if self.bool_fit['He F']:

            sample[self.map['He F']][0] = 8 * np.random.rand()
            sample[self.map['He F']][1] = np.random.rand()

            for i in range(self.n_knots['He F'] - 2):

                sample[self.map['He F']][3*i + 2] = ymax*(np.random.rand() - 0.5)
                sample[self.map['He F']][3*i + 3] = dymax*(np.random.rand() - 0.5)
                sample[self.map['He F']][3*i + 4] = d2ymax*(np.random.rand() - 0.5)

        ymax = 1
        dymax = 1
        d2ymax = 2

        if self.bool_fit['H-He p']:

            # Randomly Generate Knot Values for Rho(r)
            sample[self.map['H-He p']][0] = 0.5 + 0.5*np.random.rand()
            sample[self.map['H-He p']][1] = 1 + 0.5*(np.random.rand() - 0.5)
            sample[self.map['H-He p']][2] = 1 + 0.5*(np.random.rand() - 0.5)

            for i in range(self.n_knots['H-He p'] - 2):

                sample[self.map['H-He p']][3*i + 3] = ymax*(np.random.rand() - 0.5)
                sample[self.map['H-He p']][3*i + 4] = dymax*(np.random.rand() - 0.5)
                sample[self.map['H-He p']][3*i + 5] = d2ymax*(np.random.rand() - 0.5)

        for key in ['He-W p', 'He-H p', 'He-He p']:
    
            if self.bool_fit[key]:

                # Randomly Generate Knot Values for Rho(r)
                sample[self.map[key]][0] = 1.5 + 0.5*np.random.rand()
                sample[self.map[key]][1] = 1 + 0.5*(np.random.rand() - 0.5)
                sample[self.map[key]][2] = 1 + 0.5*(np.random.rand() - 0.5)

                for i in range(self.n_knots[key] - 2):

                    sample[self.map[key]][3*i + 3] = ymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 4] = dymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 5] = d2ymax*(np.random.rand() - 0.5)

        ymax = 4
        dymax = 10
        d2ymax = 20

        for key in ['W-He', 'He-He', 'H-He']:
            if self.bool_fit[key]:

                # Randomly Generate Knot Values for Phi(r)
                for i in range(self.n_knots[key] - 2):

                    sample[self.map[key]][3*i + 0] = ymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 1] = dymax*(np.random.rand() - 0.5)
                    sample[self.map[key]][3*i + 2] = d2ymax*(np.random.rand() - 0.5)

        if self.bool_fit['W-He p']:
            # Randomly Generate Knot Values for Phi(r)
            for i in range(self.n_knots[key] - 1):

                sample[self.map[key]][3*i + 0] = np.random.rand()
                sample[self.map[key]][3*i + 1] = np.random.rand() - 0.5
                sample[self.map[key]][3*i + 2] = np.random.rand() - 0.5

        return sample
    
    def fit_sample(self, sample):

        coef_dict = {}

        if self.bool_fit['He F']:
            
            x = np.copy(self.knot_pts['He F'])

            y = np.zeros((self.n_knots['He F'],))

            dy = np.full(y.shape, None, dtype=object)

            d2y = np.full(y.shape, None, dtype=object)

            y[-1] = 0

            dy[-1] = 0

            d2y[-1] = 0

            for i in range(self.n_knots['He F'] - 2):

                y[i + 1]   = sample[self.map['He F']][3*i + 2] 
                dy[i + 1]  = sample[self.map['He F']][3*i + 3] 
                d2y[i + 1] = sample[self.map['He F']][3*i + 4] 

            coef_dict['He F'] = splinefit(x, y, dy, d2y)

        for key in ['H-He p' ,'He-W p', 'He-H p', 'He-He p']:
            if self.bool_fit[key]:

                x = np.copy(self.knot_pts[key])

                y = np.zeros((self.n_knots[key],))

                dy = np.full(y.shape, None, dtype=object)

                d2y = np.full(y.shape, None, dtype=object)

                y[0] = 0

                y[-1] = - nexp(x[-1],  sample[self.map[key]][0], sample[self.map[key]][1], sample[self.map[key]][2])
                
                dy[-1] = - dnexp(x[-1],  sample[self.map[key]][0], sample[self.map[key]][1], sample[self.map[key]][2])

                d2y[-1] = - d2nexp(x[-1],  sample[self.map[key]][0], sample[self.map[key]][1], sample[self.map[key]][2])

                for i in range(self.n_knots[key] - 2):

                    y[i + 1]   = sample[self.map[key]][3*i + 2] 
                    dy[i + 1]  = sample[self.map[key]][3*i + 3]
                    d2y[i + 1] = sample[self.map[key]][3*i + 4]
                
                coef_dict[key] = splinefit(x, y, dy, d2y)
        

        charge = [[74, 2],[2, 2],[1, 2]]

        for i, key in enumerate(['W-He', 'He-He', 'H-He']):

            if self.bool_fit[key]:

                zbl_class = ZBL(charge[i][0], charge[i][1])
                
                x = np.copy(self.knot_pts[key])

                y = np.zeros((len(x),))

                dy = np.zeros((len(x),))

                d2y = np.zeros((len(x),))

                for i in range(self.n_knots[key] - 2):

                    y[i + 1]   = sample[self.map[key]][3*i + 0] 
                    dy[i + 1]  = sample[self.map[key]][3*i + 1]
                    d2y[i + 1] = sample[self.map[key]][3*i + 2]

                y[-1] = -zbl_class.eval_zbl(x[-1])[0]
                dy[-1] = -zbl_class.eval_grad(x[-1])[0]
                d2y[-1] = -zbl_class.eval_hess(x[-1])[0]

                coef_dict[key] = splinefit(x, y, dy, d2y)

        key = 'W-He p'

        if self.bool_fit[key]:
            
            x = np.copy(self.knot_pts[key])

            y = np.zeros((len(x),))

            dy = np.zeros((len(x),))

            d2y = np.zeros((len(x),))

            for i in range(self.n_knots[key] - 1):

                y[i]   = sample[self.map[key]][3*i + 0] 
                dy[i]  = sample[self.map[key]][3*i + 1]
                d2y[i] = sample[self.map[key]][3*i + 2]

            coef_dict[key] = splinefit(x, y, dy, d2y)

        return coef_dict
    
    def sample_to_file(self, sample):

        coef_dict = self.fit_sample(sample)
        
        rho = np.linspace(self.pot_params['rhomin'], self.pot_params['rho_c'], self.pot_params['Nrho'])

        r = np.linspace(0, self.pot_params['rc'], self.pot_params['Nr'])

        if self.bool_fit['He F']:
            
            a = abs(sample[self.map['He F']][0])
            b = abs(sample[self.map['He F']][1])

            self.pot_lammps['He F'] = np.sqrt(a**2 * rho**2 + b**2 ) - b + \
            splineval(rho, coef_dict['He F'], self.knot_pts['He F'], func = True, grad = False, hess = False)

        for key in ['H-He p' ,'He-W p', 'He-H p', 'He-He p']:
            if self.bool_fit[key]:
                self.pot_lammps[key] = nexp(r,  sample[self.map[key]][0], sample[self.map[key]][1], sample[self.map[key]][2]) + \
                    splineval(r, coef_dict[key], self.knot_pts[key], func = True, grad = False, hess = False) 

        charge = [[74, 2],[2, 2],[1, 2]]

        for i, key in enumerate(['W-He', 'He-He', 'H-He']):
            if self.bool_fit[key]:

                zbl_class = ZBL(charge[i][0], charge[i][1])
                
                zbl = zbl_class.eval_zbl(r[1:])

                poly = splineval(r[1:], coef_dict[key], self.knot_pts[key] , func = True, grad = False, hess = False)

                self.pot_lammps[key][1:] = r[1:]*(zbl + poly)

        if self.bool_fit['W-He p']:
            self.pot_lammps['W-He p'] = splineval(r, coef_dict['W-He p'], self.knot_pts['W-He p'],
                                                func = True, grad = False, hess = False) 

def sim_defect_set(optim_class:Fit_EAM_Potential):

    lmp_class = LammpsParentClass(optim_class.lammps_param, optim_class.comm, optim_class.proc_id)

    data = []

    files = glob.glob('%s/*.txt' % optim_class.lammps_folder)

    for file in files:
        
        # V(1)H(3)He(6).(8).data
        filename = os.path.basename(file)

        vac = int(filename[1])

        h = int(filename[3])

        he = int(filename[6])

        image = int(filename[8])
        
        if vac < 3:
            lmp_class.N_species = np.array([2*lmp_class.size**3 - vac, h, he])
        else:
            lmp_class.N_species = np.array([2*lmp_class.size**3 + (vac - 2), h, he])

        lmp = lammps(comm=optim_class.comm, cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])

        lmp.commands_list(lmp_class.init_from_box(pot_type='he')) 

        if os.path.getsize(file) > 0:
            xyz = np.loadtxt(file)
        else:
            xyz = np.empty((0,3))

        if xyz.ndim == 1 and len(xyz) > 0:
            xyz = xyz.reshape(1, -1)

        if vac < 3:
            for _v in range(vac):
                
                vac_pos = (2 - _v/2)*np.ones((3, ))

                lmp.command('region sphere_remove_%d sphere %f %f %f 0.1 units lattice' % 
                            (_v,vac_pos[0], vac_pos[1], vac_pos[2]))
                
                lmp.command('group del_atoms region sphere_remove_%d' % _v)
                
                lmp.command('delete_atoms group del_atoms')
                
                lmp.command('group del_atoms clear')

        if vac == 3:
            lmp.command('create_atoms 1 single 2.25 2.25 2.25 units lattice')
            lmp_class.cg_min(lmp)

        if len(xyz) > 0:
            for _x in xyz:
                lmp.command('create_atoms %d single %f %f %f units lattice' % 
                            (_x[0], _x[1], _x[2], _x[3])
                            )
        
        lmp_class.cg_min(lmp)
        
        ef = lmp_class.get_formation_energy(lmp)

        rvol = lmp_class.get_rvol(lmp)

        _data =  [vac, h, he, image, ef, rvol]
        
        data.append(_data)

    data = np.array(data)

    sort_idx = np.lexsort((data[:, 3],data[:, 2], data[:, 1], data[:, 0]))

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

def max_abs_loss(y1, y2):
    loss = 0
    for i in range(min(len(y1), len(y2))):
        loss = max( abs(y1[i] - y2[i]), loss)
    return loss

def abs_loss(y1, y2):
    loss = 0
    for i in range(min(len(y1), len(y2))):
        loss += np.abs(y1[i] - y2[i])
    return loss 

def loss_func(sample, data_ref, optim_class:Fit_EAM_Potential, diag=False, write=False, save_folder=None):

    if diag:
        t1 = time.perf_counter()    

    optim_class.sample_to_file(sample)

    loss = 0

    if optim_class.bool_fit['W-He p']:
        whe_p = optim_class.pot_lammps['W-He p']
        if (whe_p < 0).any():
            loss += 1000
            return loss

    if optim_class.bool_fit['W-He']:

        whe = optim_class.pot_lammps['W-He']

        r = np.linspace(0, optim_class.pot_params['rc'], optim_class.pot_params['Nr'])
        
        pot = whe[1:]/r[1:]

        loss += 1e-1 * np.abs(np.sum(pot[pot<0]))
        
        if diag:
            print(loss)

    if optim_class.bool_fit['He-He']:
        

        virial_coef= np.array([
        [2.47734287e+01, 5.94121916e-01],
        [2.92941502e+01, 2.40776488e+00],
        [3.07958539e+01, 3.83040639e+00],
        [3.68588657e+01, 5.40986938e+00],
        [4.17479885e+01, 6.53497823e+00],
        [4.46858331e+01, 7.17968070e+00],
        [4.75019178e+01, 8.38570392e+00],
        [5.37647405e+01, 9.02532656e+00],
        [6.15199008e+01, 9.93664731e+00],
        [6.60125239e+01, 1.03170537e+01],
        [7.25313543e+01, 1.06944122e+01],
        [8.24001392e+01, 1.14797533e+01],
        [9.07328778e+01, 1.17820755e+01],
        [1.17039231e+02, 1.21403483e+01],
        [1.41069613e+02, 1.20965893e+01],
        [1.67450895e+02, 1.21365022e+01],
        [1.93516850e+02, 1.21478229e+01],
        [2.41917917e+02, 1.21190856e+01],
        [2.67315755e+02, 1.20323657e+01],
        [2.91396089e+02, 1.19211176e+01],
        [2.68130785e+02, 1.18153354e+01],
        [3.17493260e+02, 1.16470198e+01],
        [3.69327808e+02, 1.14298383e+01],
        [4.19601366e+02, 1.11111245e+01],
        [4.67439296e+02, 1.10837355e+01],
        [5.70002943e+02, 1.08218509e+01],
        [6.68648934e+02, 1.04696549e+01],
        [7.63553410e+02, 1.01675917e+01],
        [8.72549304e+02, 9.91475627e+00],
        [1.07102569e+03, 9.29054054e+00],
        [1.26456401e+03, 8.73262548e+00],
        [1.47116726e+03, 8.23063465e+00]
        ])


        pot = optim_class.pot_lammps['He-He'][1:]

        r_pot = np.linspace(0, optim_class.pot_params['rc'], optim_class.pot_params['Nr'])[1:]

        phi = pot/r_pot

        B2_pot = eval_virial(phi, virial_coef[:, 0], r_pot)

        loss += 1e-3 * np.sum( (B2_pot - virial_coef[:, 1]) ** 2, axis = 0)

        # print('He-He Virial Loss ',  0.1 * np.sum( (B2_pot - virial_coef[:, 1]) ** 2, axis = 0))

        he_he_ref = np.array([
                        [ 1.58931000e+00,  3.28492631e-01],
                        [ 2.38396500e+00,  5.17039818e-03],
                        [ 2.64885000e+00, -3.53310542e-05],
                        [ 2.96671200e+00, -9.48768066e-04],
                        [ 3.49648200e+00, -5.38583144e-04],
                        [ 3.97327500e+00, -2.62828574e-04],
                        [ 4.76793000e+00, -8.27263709e-05]
                        ])
        
        coef_dict = optim_class.fit_sample(sample)

        zbl_class = ZBL(2, 2)

        poly = splineval(he_he_ref[:, 0], coef_dict['He-He'], optim_class.knot_pts['He-He'])

        zbl = zbl_class.eval_zbl(he_he_ref[:, 0])

        phi_pot = poly + zbl

        loss +=  np.sum((phi_pot - he_he_ref[:, 1])**2, axis=0)

        if diag:
            print('He-He Gas Loss ', loss)

    if optim_class.bool_fit['H-He']:

        h_he_ref = np.array([
        [ 1.00000000e+00,  1.53136336e+00],
        [ 1.10000000e+00,  1.14179759e+00],
        [ 1.20000000e+00,  8.52145780e-01],
        [ 1.30000000e+00,  6.35693600e-01],
        [ 1.40000000e+00,  4.73397030e-01],
        [ 1.50000000e+00,  3.51379120e-01],
        [ 1.60000000e+00,  2.59463250e-01],
        [ 1.70000000e+00,  1.90180340e-01],
        [ 1.80000000e+00,  1.38007440e-01],
        [ 1.90000000e+00,  9.88157400e-02],
        [ 2.00000000e+00,  6.94959100e-02],
        [ 2.10000000e+00,  4.76901800e-02],
        [ 2.20000000e+00,  3.15964000e-02],
        [ 2.30000000e+00,  1.98300300e-02],
        [ 2.40000000e+00,  1.13274800e-02],
        [ 2.50000000e+00,  5.27342000e-03],
        [ 2.60000000e+00,  1.04497000e-03],
        [ 2.70000000e+00, -1.83194000e-03],
        [ 2.80000000e+00, -3.71714000e-03],
        [ 2.90000000e+00, -4.88409000e-03],
        [ 3.00000000e+00, -5.53906000e-03],
        [ 3.10000000e+00, -5.83668000e-03],
        [ 3.20000000e+00, -5.89115000e-03],
        [ 3.30000000e+00, -5.78615000e-03],
        [ 3.40000000e+00, -5.58149000e-03],
        [ 3.50000000e+00, -5.32012000e-03],
        [ 3.60000000e+00, -5.03204000e-03],
        [ 3.70000000e+00, -4.73785000e-03],
        [ 3.80000000e+00, -4.45179000e-03],
        [ 3.90000000e+00, -4.18335000e-03],
        [ 4.00000000e+00, -3.93853000e-03]])
        
        coef_dict = optim_class.fit_sample(sample)

        r = np.linspace(0, optim_class.pot_params['rc'], optim_class.pot_params['Nr'])

        rho = np.linspace(optim_class.pot_params['rhomin'], optim_class.pot_params['rho_c'], optim_class.pot_params['Nrho'])

        rho_h_he = interp1d(r, optim_class.pot_lammps['H-He p'])

        rho_he_h = interp1d(r,optim_class.pot_lammps['He-H p'])

        F_h = interp1d(rho,optim_class.pot_lammps['H F'])

        F_he = interp1d(rho,optim_class.pot_lammps['He F'])

        zbl_hhe = ZBL(2, 1)

        pot_hhe = zbl_hhe.eval_zbl(h_he_ref[:,0]) + splineval(h_he_ref[:,0], coef_dict['H-He'], optim_class.knot_pts['H-He'])

        emd_H_He = np.zeros(h_he_ref[:,0].shape)
        emd_He_H = np.zeros(h_he_ref[:,0].shape)

        for i, _r in enumerate(h_he_ref[:,0]):

            _rho_h_he = rho_h_he(_r)

            emd_H_He[i] = F_he(_rho_h_he)
            
            _rho_h_he = rho_he_h(_r)

            emd_He_H[i] = F_h(_rho_h_he)
        
        pairwise = (emd_H_He + emd_He_H + pot_hhe)

        loss += 1e-1 * np.sum((1 - pairwise/h_he_ref[:, 1])**2, axis=0)
        
        if diag:
            print('H-He Gas Loss: ', loss)


    write_pot(optim_class.pot_lammps, optim_class.potlines, optim_class.lammps_param['potfile'])

    data_sample = sim_defect_set(optim_class)
 
    ref_mat = lst2matrix(data_ref)

    sample_mat = lst2matrix(data_sample)


    ''' 

    Loss from Helium Interstitial
    
    Image 0: Tet
    Image 1  Random Site - which optimizes to Tet
    Image 2: <110>
    Image 3: Oct
    Image 5: <111> 
    
    '''
    # Loss due to difference in Tet Formation Energy
    loss += 5 * np.abs(sample_mat[0, 0, 1, 0, 0] - ref_mat[0, 0, 1, 0, 0]) ** 2

    loss += rel_abs_loss(sample_mat[0, 0, 1, 1:, 0] - sample_mat[0, 0, 1, 0, 0], ref_mat[0, 0, 1, 1:, 0] - ref_mat[0, 0, 1, 0, 0])

    loss += 5 * abs(1 - (sample_mat[0, 0, 1, 2, 0] - sample_mat[0, 0, 1, 0, 0])/(ref_mat[0, 0, 1, 2, 0] - ref_mat[0, 0, 1, 0, 0]) )

    # Loss due to difference in Relaxation Volume
    loss += np.abs(1 - sample_mat[0, 0, 1, 0, 1]/ref_mat[0, 0, 1, 0, 1])

    loss += np.abs(1 - sample_mat[0, 0, 1, 3, 1]/ref_mat[0, 0, 1, 3, 1])
    
    if diag:
        print(sample_mat[0, 0, 1, :, :], ref_mat[0, 0, 1, :, :])
    # print(sample_mat[0, 0, 1, 1:, 0] - sample_mat[0, 0, 1, 0, 0], ref_mat[0, 0, 1, 1:, 0] - ref_mat[0, 0, 1, 0, 0])
    ''' Constraint '''

    constraint = not (np.arange(sample_mat.shape[3]) == np.round(sample_mat[0, 0, 1, :, 0], 3).argsort()).all()
    
    loss += 100*constraint  

    constraint = not len(np.unique(np.round(sample_mat[0, 0, 1, :, 0], 2))) == sample_mat.shape[3]

    loss += 100*constraint  

    if sample_mat.shape[2]  > 2:
        loss += rel_abs_loss(sample_mat[0, 0, 2, 1:, 0] - sample_mat[0, 0, 1, 0, 0], ref_mat[0, 0, 2, 1:, 0] - ref_mat[0, 0, 1, 0, 0])

    ''' 
    Loss from He-He Binding

    Interstital 
    Vacancy 
    Di-Vacancy

    '''
    for v in range(sample_mat.shape[0]):
            
        binding_sample = subtract_lst(np.min(sample_mat[v, 0, 1:, :, 0], axis = 1), np.min(sample_mat[v, 0, :-1, :, 0], axis = 1))
        
        binding_sample = sample_mat[0, 0, 1, 0, 0] - binding_sample

        binding_ref = subtract_lst(np.min(ref_mat[v, 0, 1:, :, 0], axis = 1), np.min(ref_mat[v, 0, :-1, :, 0], axis = 1))
        
        binding_ref = ref_mat[0, 0, 1, 0, 0] - binding_ref

        if v == 0:
            loss += 5 * rel_abs_loss(binding_sample, binding_ref)
        elif v == 3:
            loss += 5 * rel_abs_loss(binding_sample, binding_ref)
        else:
            loss += 1 * rel_abs_loss(binding_sample, binding_ref)
        if diag:
            print(v, 0 ,binding_sample, binding_ref, loss)

    '''
    Loss from H-He Binding

    Interstital 
    Vacancy 
    Di-Vacancy

    '''
    for v in range(sample_mat.shape[0]):

        for h in range(1, min(sample_mat.shape[1], ref_mat.shape[1])):

            binding_sample = subtract_lst(np.min(sample_mat[v, h, :, :, 0], axis = 1), np.min(sample_mat[v, h-1, :, :, 0], axis = 1))
            
            binding_sample = sample_mat[0, 1, 0, 0, 0] - binding_sample

            binding_ref = subtract_lst(np.min(ref_mat[v, h, :, :, 0], axis = 1), np.min(ref_mat[v, h-1, :, :, 0], axis = 1))
            
            binding_ref = ref_mat[0, 1, 0, 0, 0] - binding_ref

            if v == 1:
                loss += 10 * rel_abs_loss(binding_sample, binding_ref)
            else:
                loss += 1 * rel_abs_loss(binding_sample, binding_ref)

            if diag:
                print( v, h ,binding_sample, binding_ref, loss )


    ''' Loss from Relaxation Volumes '''

    for i in range(min(sample_mat.shape[0], ref_mat.shape[0])):
        for j in range(min(sample_mat.shape[1], ref_mat.shape[1])):
            for k in range(min(sample_mat.shape[2], ref_mat.shape[2])):
                for l in range(min(sample_mat.shape[3], ref_mat.shape[3])):

                    r_sample = sample_mat[i, j, k, l, 1]

                    r_ref = ref_mat[i, j, k, l, 1]

                    if not (np.isinf(r_ref) or np.isinf(r_sample)):
                        loss += 5 * abs(r_sample - r_ref)
    if diag:
        t2 = time.perf_counter()
        
        print(sample,loss, t2 - t1)

    if write:

        if loss < 20:
            with open(os.path.join(save_folder, 'Loss_%d.txt' % optim_class.proc_id), 'a') as file:
                file.write('%f \n' % loss)
        
            with open(os.path.join(save_folder, 'Samples_%d.txt' % optim_class.proc_id), 'a') as file:
                string = ''
                for _x in sample:
                    string += '%.4f ' % _x
                string += '\n'

                file.write(string)
        # print(sample, loss)
    return loss


def random_sampling(n_knots, comm, proc_id, max_time=3, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')

    # pot_params['rho_c'] = (pot_params['Nrho'] - 1)*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_yang.txt')

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


def loss_w_he_lj(x, eam_fit, ref):

    loss = 0

    coef_dict = eam_fit.fit_sample(x)

    zbl_class = ZBL(2, 74)

    poly = splineval(ref[:, 0], coef_dict['W-He'], eam_fit.knot_pts['W-He'])

    zbl = zbl_class.eval_zbl(ref[:, 0])

    phi_pot = poly + zbl
    
    loss = np.sum((phi_pot - ref[:, 1])**2, axis=0)

    return loss

def min_w_he_lj(x):
    
    n_knots = {}
    n_knots['He F'] = 0
    n_knots['H-He p'] = 0
    n_knots['He-W p'] = 0
    n_knots['He-H p'] = 0
    n_knots['He-He p'] = 0
    n_knots['W-He'] = 4
    n_knots['He-He'] = 0
    n_knots['H-He'] = 0

    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')

    eam_fit = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, None, 0, '')

    x = np.linspace(2, 4, 20)

    lj = np.array([5e-3, 2.50])

    y = 4 * lj[0] * ( (lj[1]/x)**12 - (lj[1]/x)**6 )

    ref = np.column_stack([x, y])

    sample = np.hstack([-2,  2 ,-1, -0.2,  0.5, -1])

    res = minimize(loss_w_he_lj, sample, args=(eam_fit, ref), method='Powell', options={'maxfev':1e4})

    return res.x

def lj_sampling(n_knots, comm, proc_id, mean, cov, max_time=3, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')

    # pot_params['rho_c'] = (pot_params['Nrho'] - 1)*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_yang.txt')

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

        x = np.zeros(mean.shape)

        for i in range(len(mean)):

            x[i] = np.random.normal(loc=mean[i], scale=cov[i], size=1)

        whe_sample = min_w_he_lj(x[-2:])

        sample = np.hstack([x[:-2], whe_sample])

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


def gaussian_sampling(n_knots, comm, proc_id, mean, cov, max_time=3, work_dir = '../Optim_Local', save_folder = '../Fitting_Output'):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)

    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')

    # pot_params['rho_c'] = (pot_params['Nrho'] - 1)*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_yang.txt')

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



def simplex(n_knots, comm, proc_id, x_init, maxiter = 100, work_dir = '../Optim_Local', save_folder = '../Fitting_Output', diag=False):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)


    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')
    # pot, potlines, pot_params = read_pot('Fitting_Runtime/Potentials/optim.0.eam.he' )

    # pot_params['rho_c'] = (pot_params['Nrho'] - 1)*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_yang.txt')

    # Init Optimization Parameter
    t1 = time.perf_counter()

    _ = loss_func(x_init, data_ref, fitting_class)

    t2 = time.perf_counter()

    if proc_id == 0:
        print('Average Time: %.2f s' % (t2 - t1))
        sys.stdout.flush()    
    
    res = minimize(loss_func, x_init, args=(data_ref, fitting_class, diag, False, None), method='Powell',
                   options={"maxfev":maxiter, "return_all":True}, tol=1e-4)

    res.allvecs = np.array(res.allvecs)
    
    with open(os.path.join(save_folder, 'Samples_%d.txt' % proc_id), 'a') as file:
        np.savetxt(file, res.allvecs, fmt='%.4f')

    with open(os.path.join(save_folder, 'Loss_%d.txt' % proc_id), 'a') as file:
        file.write('%.4f\n' % res.fun)

    with open(os.path.join(save_folder, 'Optima_%d.txt' % proc_id), 'a') as file:
        np.savetxt(file, np.array([res.x]), fmt='%.4f')
    # local_minimizer = {
    #     'method': 'BFGS',
    #     'args': (data_ref, fitting_class, True),
    #     'options': {"maxiter": 20},
    #     'tol': 1e-4
    # }
    # res = basinhopping(loss_func, x_init, minimizer_kwargs=local_minimizer, niter=maxiter)

    return res.x


def genetic_alg(n_knots, comm, proc_id, work_dir = '../Optim_Local', save_folder = '../Fitting_Output', diag=False, write=True):

    data_files_folder = os.path.join(work_dir, 'Data_Files')

    lammps_folder = os.path.join(work_dir, 'Data_Files_%d' % proc_id)

    if os.path.exists(lammps_folder):

        shutil.rmtree(lammps_folder)
    
    shutil.copytree(data_files_folder, lammps_folder)


    # Read Daniel's potential to initialize the W-H potential and the params for writing a .eam.he file
    pot, potlines, pot_params = read_pot('git_folder/Potentials/init.eam.he')
    # pot, potlines, pot_params = read_pot('Fitting_Runtime/Potentials/optim.0.eam.he' )

    # pot_params['rho_c'] = (pot_params['Nrho'] - 1)*pot_params['drho']
    
    # Call the main fitting class
    fitting_class = Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, work_dir)

    data_ref = np.loadtxt('dft_yang.txt')

    # Init Optimization Parameter
    t1 = time.perf_counter()

    _ = loss_func(fitting_class.gen_rand(), data_ref, fitting_class)

    t2 = time.perf_counter()

    if proc_id == 0:
        print('Average Time: %.2f s' % (t2 - t1))
        sys.stdout.flush()    
    
    color = proc_id % 8

    bounds = (
               (color, color + 1), (0, 1),
               (1, 100), (1, 10),
               (-3, -0.5), (1, 5), (-2, 5), (-1, 0.2), (-1, 2), (-3, 3),
               (0.2, 1), (-1, 0), (-0.1, 0.1), (0, 0.1), (-0.2, 0), (-0.1, 0.1)
              )       
    
    res = differential_evolution(loss_func, bounds = bounds, args=(data_ref, fitting_class, diag, write, save_folder), popsize=50)

    # local_minimizer = {
    #     'method': 'BFGS',
    #     'args': (data_ref, fitting_class, True),
    #     'options': {"maxiter": 20},
    #     'tol': 1e-4
    # }

def gmm(file_pattern, data_folder, iter):
    loss_lst = []  

    nfiles = len(glob.glob(os.path.join(file_pattern, 'Filtered_Loss_*.txt')))
                         
    # Load loss data from files
    for i in range(nfiles):
        file  = os.path.join(file_pattern, 'Filtered_Loss_%d.txt' % i)
    
        if os.path.getsize(file) > 0:
            loss_lst.append(np.loadtxt(file))
    
    if not loss_lst:
        raise ValueError("No loss data found.")
    
    loss = np.hstack([x for x in loss_lst])

    sample_lst = []  

    # Load sample data from files
    for i in range(nfiles):
        file  = os.path.join(file_pattern, 'Filtered_Samples_%d.txt' % i)

        if os.path.getsize(file) > 0:
            sample_lst.append(np.loadtxt(file))
    
    if not sample_lst:
        raise ValueError("No sample data found.")
    
    samples = np.vstack([x for x in sample_lst])

    print("Mean Loss:", loss.mean(), "Min Loss:", loss.min())

    # Sort samples based on loss values
    sort_idx = np.argsort(loss)
    loss = loss[sort_idx]
    samples = samples[sort_idx]

    # Select samples with loss less than twice the minimum loss
    thresh_idx = np.where(loss < 2 * loss.min())[0]

    # Limit the number of samples to a maximum of 10000
    n = np.clip(10000, a_min=0, a_max=len(thresh_idx)).astype(int)
    print("Threshold indices and number of samples:", thresh_idx, n)
    
    data = samples[thresh_idx[:n]]

    # Fit GMM and determine the optimal number of components using BIC
    cmp = 1
    gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=1e-6)
    gmm.fit(data)
    bic_val = gmm.bic(data)
    bic_val_prev = bic_val

    print("Initial components and BIC:", cmp, bic_val)
    sys.stdout.flush()  

    while True:
        cmp += 1
        bic_val_prev = bic_val
        gmm = GaussianMixture(n_components=cmp, covariance_type='full', reg_covar=4e-3)
        gmm.fit(data)
        bic_val = gmm.bic(data)
        print("Components:", cmp, "BIC:", bic_val, "Previous BIC:", bic_val_prev)

        if 1.01 * bic_val > bic_val_prev:
            break

    final_components = cmp - 1
    print("Optimal number of components:", final_components)
    sys.stdout.flush()

    gmm = GaussianMixture(n_components=final_components, covariance_type='full')
    gmm.fit(data)

    print("Data shape:", data.shape)
    print("GMM means:", gmm.means_)

    # Create directory for GMM results if it doesn't exist
    gmm_folder = os.path.join(data_folder, f'GMM_{iter}')
    os.makedirs(gmm_folder, exist_ok=True)

    # Save filtered loss, samples, and GMM parameters
    np.savetxt(os.path.join(gmm_folder, 'Filtered_Loss.txt'), loss[thresh_idx[:n]])
    np.savetxt(os.path.join(gmm_folder, 'Filtered_Samples.txt'), data)
    np.save(os.path.join(gmm_folder, 'Cov.npy'), gmm.covariances_)
    np.save(os.path.join(gmm_folder, 'Mean.npy'), gmm.means_)

    return gmm.means_, gmm.covariances_