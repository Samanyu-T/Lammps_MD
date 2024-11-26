import sys
import os
import json
import numpy as np
import shutil
sys.path.append(os.path.join(os.getcwd(), 'git_folder', 'Classes'))
from lammps import lammps
import matplotlib.pyplot as plt
import math
import He_Fitting, Handle_PotFiles_He
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

def sim_h_he(r, potfile,type='he'):
    lmp = lammps( cmdargs=['-screen', 'none', '-echo', 'none', '-log', 'none'])
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')
    lmp.command('boundary p p p')
    lmp.command('lattice fcc 20')
    lmp.command('region r_simbox block 0 1 0 1 0 1 units lattice')
    lmp.command('region r_atombox block 0 1 0 1 0 1 units lattice')
    lmp.command('create_box 3 r_simbox')
    lmp.command('create_atoms 3 single 0 0 0')
    lmp.command('create_atoms 2 single 0 0 %f units box' % r)
    lmp.command('mass 1 183.84')
    lmp.command('mass 2 1.00784')
    lmp.command('mass 3 4.002602')
    lmp.command('pair_style eam/%s' % type )
    lmp.command('pair_coeff * * %s W H He' % potfile)
    lmp.command('thermo_style custom step temp pe pxx pyy pzz pxy pxz pyz vol')
    lmp.command('thermo 100')
    lmp.command('run 0')
    lmp.command('write_dump all custom dump.%i.atom id type x y z' % i)
    pe = lmp.get_thermo('pe')
    
    return pe

def analytical_h_he(x, eam_fit, data_dft):

    # x = np.hstack([0.05, 0.06, x])
    eam_fit.sample_to_file(x)

    r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])
    rho = np.linspace(eam_fit.pot_params['rhomin'], eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho'])

    rho_h_he = interp1d(r, eam_fit.pot_lammps['H-He p'])

    rho_he_h = interp1d(r,eam_fit.pot_lammps['He-H p'])

    F_h = interp1d(rho,eam_fit.pot_lammps['H F'])

    F_he = interp1d(rho,eam_fit.pot_lammps['He F'])

    zbl_hhe = He_Fitting.ZBL(2, 1)

    r_plt = data_dft[:,0]

    coef_dict = eam_fit.fit_sample(x)

    pot_hhe = zbl_hhe.eval_zbl(r_plt) + He_Fitting.splineval(r_plt, coef_dict['H-He'], eam_fit.knot_pts['H-He'])

    emd_H_He = np.zeros(r_plt.shape)
    emd_He_H = np.zeros(r_plt.shape)

    for i, _r in enumerate(r_plt):

        _rho_h_he = rho_h_he(_r)

        emd_H_He[i] = F_he(_rho_h_he)
        
        _rho_h_he = rho_he_h(_r)

        emd_He_H[i] = F_h(_rho_h_he)

    total_hhe = (emd_H_He + emd_He_H + pot_hhe)

    pks, _ = find_peaks(pot_hhe)

    loss = 0

    loss += len(pks)

    k = 0

    loss += np.linalg.norm( (total_hhe[k:] - data_dft[k:,1]) )
    
    if x[0] < 1:
        loss += 10
    print(x, loss)

    return loss

def loss_func(x, eam_fit, data_dft):

    Zh = 1
    Zhe = 2
    A = 5.5
    h = 0.9
    a0 = 0.529
    k = 0.1366
    # x = np.hstack([A, Zh, x[0], Zhe, x[1], x[2:]])
    loss = 1e6 *( (x[0] > 1) + (x[2] > 2) )

    x = np.hstack([A, 0.6, x])

    eam_fit.sample_to_file(x)

    Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])


    for i, row in enumerate(data_dft):
        r, dft_pe = row
    
        pot_pe = sim_h_he(r, eam_fit.lammps_param['potfile'])

        loss += (1 - pot_pe/dft_pe)**2

    print(x, loss)
    return loss

data_dft = []

with open('hhe_energy.dat', 'r') as file:
    for line in file:
        split = [txt for txt in line.split(' ') if txt != '']
        r =  float(split[0][-3:])
        pe = float(split[-1])
        data_dft.append([r, pe])
data_dft = np.array(data_dft)

r = np.linspace(1.5, 4, 100)
zbl = He_Fitting.ZBL(2, 1)
y = zbl.eval_zbl(r)

# data_dft = np.hstack([r.reshape(-1,1), y.reshape(-1,1)])
comm = 0

proc_id = 0

n_procs = 1

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/init.eam.he')


n_knots = {}
n_knots['He F'] = 0
n_knots['H-He p'] = 0
n_knots['He-W p'] = 0
n_knots['He-H p'] = 0
n_knots['He-He p'] = 0
n_knots['W-He'] = 0
n_knots['He-He'] = 0
n_knots['H-He'] = 4
n_knots['W-He p'] = 0

with open('fitting.json', 'r') as file:
    param_dict = json.load(file)

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])


# x = np.array([-1.875e-01,  2.568e-01, -2.578e-01, -1.852e-02,  2.622e-02, -3.972e-02])


x = np.array([1.10116422, -0.17191936,  0.37648485, -2.697355,  1.80101015, -0.1034058, 0.07422518, -0.01610268])

print(eam_fit.gen_rand().shape, x.shape)

eam_fit.sample_to_file(x)
r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]
hhe = eam_fit.pot_lammps['H-He'][1:]
hhe = hhe / r
plt.plot(r[400:], hhe[400:])
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')
# -0.049354940365810  ,-0.000156441433979  ,-0.009534396608074  ,2.850787994948958  ,0.006478313126762  ,0.001373440533456  ,-0.001456154165614  ,

x = np.array([                1.029187717904903  ,-0.230617861727848  ,0.516418837384891  ,-3.563055960429090  ,1.976148194654602  ,-0.104694805988491  ,0.051599506311914  ,-0.027733679796375  ,

    # 1.042229043335329  ,-0.340770793334825  ,0.519314705977125  ,-2.927848634055453  ,2.035806391796232  ,-0.117687562350735  ,0.123935541628657  ,-0.092381078856146  ,#                1.07444089, -0.32392346 , 0.82255069, -3.88500038,  1.73938636, -0.11506669, 0.10185475, -0.08290284,
])
# 1.062806371993989  ,-0.396125345216717  ,0.801932928188837  ,-3.396647583692781  ,1.845726226191536  ,-0.122792356884467  ,0.159264319354548  ,-0.232146198186701  ,])
            #  -1.12250698e-01,  3.95622407e-02,  1.49297332e-01, -2.38165659e-02, 2.79419759e-02, -5.00556693e-02])

# x = np.random.randn(6)

# x_res = minimize(analytical_h_he, x, args=(eam_fit, data_dft), method='Powell',options={"maxfev":10000}, tol=1e-4)
# print(x_res)
# x = x_res.x

# x = np.hstack([0.05, 0.06, x])

eam_fit.sample_to_file(x)

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/init.eam.he')




# r_plt = np.linspace(0.5, 4, 100)
# zbl = He_Fitting.ZBL(2, 1)
# y = zbl.eval_zbl(r)

r_plt = data_dft[:, 0]

pe_arr = np.zeros((len(r_plt,)))

for i, _r in enumerate(r_plt):
    pot_pe = sim_h_he(_r, eam_fit.lammps_param['potfile'])
    
    pe_arr[i] = pot_pe

h_he_ref = np.array([
            [ 2.64885000e+00,  5.92872325e-03],
            [ 2.91373500e+00,  1.38739018e-03],
            [ 3.17862000e+00, -3.86056397e-04],
            [ 3.44350500e+00, -5.48062207e-04],
            [ 3.70839000e+00, -5.85978460e-04],
            [ 3.97327500e+00, -4.22249185e-04],
            [ 4.23816000e+00, -3.75715601e-04],
            [ 4.76793000e+00, -1.68037941e-04],
            ])

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]

zbl = He_Fitting.ZBL(2, 1)
plt.plot(r_plt, pe_arr, label='full inc eam')
plt.plot(data_dft[:,0], zbl.eval_zbl(data_dft[:,0]), label = 'zbl')  
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')

pot = eam_fit.pot_lammps['H-He'][1:]/r
plt.plot(r[125:], pot[125:], label='pairwise component')

plt.scatter(h_he_ref[:,0], h_he_ref[:,1], label='qmc', color='red')

plt.xlabel('Lattice Constant/ A')
plt.ylabel('Energy/ eV')
plt.title('Vacuum Interaction of H-He')
plt.legend()
plt.show()

r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])
rho = np.linspace(eam_fit.pot_params['rhomin'], eam_fit.pot_params['rho_c'], eam_fit.pot_params['Nrho'])

rho_h_he = interp1d(r, eam_fit.pot_lammps['H-He p'])

rho_he_h = interp1d(r,eam_fit.pot_lammps['He-H p'])

F_h = interp1d(rho,eam_fit.pot_lammps['H F'])

F_he = interp1d(rho,eam_fit.pot_lammps['He F'])

zbl_hhe = He_Fitting.ZBL(2, 1)

r_plt = data_dft[:,0]

coef_dict = eam_fit.fit_sample(x)

pot_hhe = zbl_hhe.eval_zbl(r_plt) + He_Fitting.splineval(r_plt, coef_dict['H-He'], eam_fit.knot_pts['H-He'])

emd_H_He = np.zeros(r_plt.shape)
emd_He_H = np.zeros(r_plt.shape)

for i, _r in enumerate(r_plt):

    _rho_h_he = rho_h_he(_r)

    emd_H_He[i] = F_he(_rho_h_he)
    
    _rho_h_he = rho_he_h(_r)

    emd_He_H[i] = F_h(_rho_h_he)

total_hhe = (emd_H_He + emd_He_H + pot_hhe)
np.savetxt('h_he_pairwise.txt',data_dft)
plt.plot(r_plt, total_hhe, label='our-work')
plt.scatter(data_dft[:,0], data_dft[:,1], label='DFT', color='black', marker='o')
# plt.plot(r_plt, emd_H_He, label='Electron density of Hydrogen on Helium')
# plt.plot(r_plt, emd_He_H, label='Electron density of Helium on Hydrogen')
# plt.plot(r_plt, pot_hhe, label='Covalent')

plt.ylabel('Interaction Energy / eV', fontsize=16)
plt.xlabel('Radial Distance/ A', fontsize=16)
plt.title('H-He Pairwise Interaction', fontsize=16)

np.savetxt('r_hhe.txt', r_plt)
np.savetxt('our-hhe.txt', total_hhe)


np.savetxt('dft_hhe.txt', data_dft)

plt.legend(fontsize=14)
plt.show()

plt.plot(eam_fit.pot_lammps['H-He p'])
plt.plot(eam_fit.pot_lammps['He-H p'])
plt.show()

plt.plot(rho, eam_fit.pot_lammps['He F'])
plt.show()



n_knots = {}
n_knots['He F'] = 2
n_knots['H-He p'] = 3
n_knots['He-W p'] = 3
n_knots['He-H p'] = 3
n_knots['He-He p'] = 3
n_knots['W-He'] = 4
n_knots['He-He'] = 4
n_knots['H-He'] = 4
n_knots['W-He p'] = 3

pot, potlines, pot_params = Handle_PotFiles_He.read_pot('git_folder/Potentials/beck_full.eam.he')

eam_fit = He_Fitting.Fit_EAM_Potential(pot, n_knots, pot_params, potlines, comm, proc_id, param_dict['work_dir'])



# x = np.array([ 2.25309532e+00,  5.78508321e-01,
#                1.36606702e+00,  4.35316203e+00, 1.00000000e-03,
#                4.78434000e+01,  6.79830000e+00, 1e-3,
#                -2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
#                 0, 0, 0,
#               -8.78443579e-01,  1.48624430e+00,  2.27126353e+00, -3.40994311e-01,  5.90983069e-01, -2.18796556e-01,
#               -3.67006528e-01,  4.78912258e-01, -3.76192674e-01, -2.75952106e-02,  4.34246773e-02, -7.47000749e-02,
#               -1.12250698e-01,  3.95622407e-02,  1.49297332e-01, -2.38165659e-02, 2.79419759e-02, -5.00556693e-02])
x = np.array([   1.7015e+00,  4.2490e-01, 
                 0, 0, 0, 2.7236, 0, 0, 0,
                 0, 0, 0, 2.7236, 0, 0, 0,
                 0, 0, 0, 2.7236, 0, 0, 0,
                 0, 0, 0, 2.7236, 0, 0, 0,
                 1.7581, -1.1982e+00,  3.1443e+00, -3.2970e-01, 2.7236, -2.2820e-01,  4.1590e-01, -4.7750e-01 ,
                 1.61712964, -3.670e-01,  4.789e-01 ,-3.762e-01,  3.23425928, -2.760e-02,  4.344e-02, -7.470e-02, 
                 1.00000043, -0.21809812,  0.61560816, -3.40915359,  1.8661813,  -0.09889802, 0.07572234, -0.02700178,
                 6.8130e-01, -3.8090e-01,  6.3500e-02,   3.27332980, 8.6000e-03,  -9.4000e-03, 1.3100e-02])

                # He-F
x = np.array([  
    #  5.938102944985967  ,0.487301957901169 ,
# 7.152587706307302  ,0.679913418219516  ,
# 7.151316759081570  ,0.679809511741090  ,
7.173175729203135  ,0.677984683567253  ,

                # H-He p
# -0.112164251964737  ,0.246676734466499  ,0.006932113511024  ,2.150515088760416  ,-0.006099785108573  ,0.059926654082773  ,-0.041175958794221  ,
                # -0.229293483153587  ,0.267741256392708  ,-0.601124316643698  ,1.678195297921719  ,-0.013373592975755  ,-0.055811205993631  ,-0.031165542222783  ,
# -0.111236335417731  ,0.256971029685829  ,0.005931181470555  ,2.449068606554940  ,-0.006040726496581  ,0.072763303679693  ,-0.038961667283644  ,
# -0.150774277978756  ,0.030606843989373  ,0.007805107197474  ,2.535886704460149  ,-0.007904053988881  ,0.066252766663477  ,-0.035402711186644  ,
# -0.154386864213040  ,0.035755954300066  ,0.008953780698048  ,2.242465710462394  ,-0.009327623090433  ,0.024724887548620  ,-0.040842045425463  ,
# -0.154456742019765  ,0.035828032183672  ,0.008896232060785  ,2.242703901724083  ,-0.009074411339400  ,0.023843474660805  ,-0.046286280219683  ,
# -0.137769963091770  ,0.036403494370654  ,0.004566444248888  ,2.216382596817530  ,-0.006825932535634  ,0.031348799435969  ,-0.088072887610813  ,
# 0.067118806847826  ,0.036381952295258  ,0.004566444248888  ,2.216382596817530  ,-0.006825932535634  ,0.031348799435969  ,-0.088072887610813  ,
# 0.019166671629807  ,0.028341694828852  ,0.001008662672794  ,2.898053462139851  ,-0.006816647148183  ,0.030833535225772  ,-0.088266819954073  ,

0.019165297145552  ,0.028468950721278  ,0.001009769713273  ,2.896548043654880  ,-0.006809073565067  ,0.030726973927345  ,-0.090300463262235  ,
# -0.133140401351112  ,0.038907552276616  ,0.004770050379142  ,1.898887165743574  ,-0.007070557661200  ,0.031812660420496  ,-0.089912138479008  ,
# -0.138964431888005  ,0.039180075871076  ,0.004971549999070  ,1.881886460937327  ,-0.006576878352097  ,0.035131145915772  ,-0.086409528214927  ,
                # -0.139610613100625  ,0.029021903994268  ,0.004830487166574  ,2.984126212950756  ,-0.008150868478041  ,0.029163048259176  ,-0.078207375511427  ,
                # -0.079128452408058  ,0.044994808778542  ,0.013499237887564  ,2.988793913103802  ,-0.006773316255717  ,0.045196153094349  ,-0.075494663653198  ,
                # -0.108535186962440,   0.041131066515649,   0.005442635288297 ,  2.539717724596742 ,  -0.006095049742697 ,  0.040951644597138 ,  -0.094057665968566,
                # -0.049354940365810  ,-0.000156441433979  ,-0.009534396608074  ,2.850787994948958  ,0.006478313126762  ,0.001373440533456  ,-0.001456154165614  ,
                # He-W p
                #  0.57890303, -0.56686309, -0.07219475,  2.144805,   -0.030681,   -0.01471756, 0.25942299,
# 1.358644945113586  ,-1.829837502089230  ,1.112976311239351  ,2.148674619763379  ,-0.032434921984025  ,-0.012792084965765  ,0.275668083566939  ,
# 1.353807395966658  ,-1.822016680374136  ,1.099901027859676  ,2.140451907695307  ,-0.035075816739765  ,-0.012395440239504  ,0.283085805319375  ,
# 1.598149660786222  ,-1.168725636196623  ,1.358327548739671  ,2.172891041031724  ,-0.032417263073891  ,-0.013980427783148  ,0.292028718371396  ,
#   6.09680502e-01, -4.48519160e-01, -3.21876141e-05,  2.13613874e+00, -3.07222500e-02, -1.31348900e-02,  2.59788020e-01,
# 1.353812737507029  ,-1.822818501387960  ,1.124554730912103  ,2.141294949168543  ,-0.035090993813126  ,-0.012401505547773  ,0.283198260527466  ,
# -0.515636377332946  ,0.590139012937681  ,0.167224075380608  ,2.113956869783618  ,-0.030689292460099  ,-0.012007213192311  ,0.265543435376348  ,
# 1.337965970443441  ,-1.877017180922142  ,1.137219797129565  ,2.131301849113787  ,-0.035452339592472  ,-0.013720344639032  ,0.266539142813989  ,
# 1.599607089963630  ,-2.107225706082976  ,0.891271311174638  ,2.046121087684851  ,-0.029713851729303  ,-0.014790041697685  ,0.197709365122589  ,
# 1.622562420153173  ,-2.120696552493505  ,0.907770987713341  ,2.036483199848319  ,-0.028554043838931  ,-0.014809887020939  ,0.203827416009612  ,
# 1.684483463110801  ,-2.037065140587740  ,0.881738085927047  ,2.026552098856636  ,-0.025066716977647  ,-0.015514116010266  ,0.216375882888180  ,
# 1.669730325961347  ,-2.155881000269711  ,0.905095758308033  ,1.998058625636716  ,-0.030281413284978  ,-0.014629966357746  ,0.200609397531732  ,
# 1.601164801079278  ,-2.106954418966872  ,0.894376930640958  ,2.046619393351958  ,-0.029743411079993  ,-0.014790038071384  ,0.197582675627487  ,
# 1.603579432411482  ,-2.106839870384734  ,0.896578392759885  ,2.050593969249934  ,-0.029809157954285  ,-0.015399003364018  ,0.204539494482383 
# 1.602297521078375  ,-2.107204269313120  ,0.896799407939159  ,2.049811635355677  ,-0.029819010763987  ,-0.015391100314918  ,0.204530591946528,
# 1.608260367305212  ,-2.110774204721858  ,0.944473806230391  ,2.048033103857013  ,-0.029691750383379  ,-0.015444634878264  ,0.204857986476913  ,    
1.665735077028031  ,-2.137269970890688  ,0.953578648138977  ,2.002909488847890  ,-0.029523143157695  ,-0.015859721302153  ,0.199783375977483  ,
                # He-H p
                #  0.219373054974  ,-0.000571176308  ,0.000008464621  ,2.723670801389  ,-0.001189548076  ,-0.000000027764  ,0.000000000065  ,
                # 0, 0, 0, 2, 0, 0, 0, 
                # -0.000422939150  ,-0.000001163407  ,0.000000021786  ,2.000000000000  ,0.000000000000  ,0.000000000000  ,0.000000000000  ,
                # 0.000027733042348 ,  -0.001664991673427,   0.000006825345551,   1.563764062651298,   0.000037361837587,   0.000120091598099,   0.000219578142364,
-0.001122660787652  ,-0.000455723657045  ,-0.000563826760888  ,2.080935848157005  ,-0.000281006077076  ,-0.000000000195180  ,-0.001801416605330  ,
# -0.033702475352507  ,-0.000282103449437  ,-0.000028110009367  ,2.145898025156000  ,0.000000000000000  ,0.000000000000000  ,0.000000000000000  ,
                
                
                #  He-He p
                #  0.594690952901  ,-0.003014856809  ,-0.000185516565  ,2.000000166982  ,0  ,0  ,0,
                #  0.185479465811  ,0.008470616936  ,0.001077780079  ,1.999949607675  ,0.000000000000  ,0.000000000000  ,0.000000000000  ,
                # 0.383091837975847  ,0.009079475509070  ,0.001077780079000  ,1.999949607675000  ,0.000000000000000  ,0.000000000000000  ,0.000000000000000  ,
                 0, 0, 0, 2, 0, 0,  0,


                #  W-He
                #  1.5115436068  ,-1.4591796094  ,4.0439931377  ,2.1100374932  ,2.2042056669  ,-0.4158906200  ,0.3796934500  ,-0.0281096000  ,               
                 #  1.50669244, -1.21951906,  3.76651107,  2.07499888 , 2.20411374 ,-0.42493129, 0.37492672, -0.02326983,              
# 1.515260479051914  ,-1.383901949412876  ,4.199404020275592  ,2.178293267982525  ,2.220831490176620  ,-0.420712428952096  ,0.370717302423405  ,-0.030153911079322  ,
# 1.511298768711221  ,-1.397758019862870  ,4.209294655713867  ,2.192846162938463  ,2.218686901217146  ,-0.410678069340218  ,0.391124554001964  ,-0.029884520087698  ,
# 1.532316248214989  ,-1.440581571829140  ,4.238503501734657  ,2.068639120921741  ,2.122720559493460  ,-0.439000266995463  ,0.386000592776747  ,-0.030294788820395  ,
# 1.510737716789883  ,-1.409649567374363  ,4.207864858669675  ,2.189261922297460  ,2.223881881456646  ,-0.411205282540836  ,0.390960832979292  ,-0.031041785281362  ,
# 1.650313364200363  ,-1.156477127448256  ,3.998346852344760  ,2.174848221701171  ,2.096940695454058  ,-0.452684510472797  ,0.414342318607972  ,-0.033077973422009  ,
# 1.509057842634742  ,-1.431651441936010  ,4.163756796589393  ,2.198419216093388  ,2.204143325793484  ,-0.408577085741108  ,0.395311067089667  ,-0.031363996769826  ,
# 1.506439764088560  ,-1.451930436688668  ,4.061642450010652  ,2.184566173677435  ,2.185425432143323  ,-0.424752496420229  ,0.382663269118793  ,-0.033491396364589  ,
# 1.503814306001888  ,-1.455497361444964  ,4.072004576575942  ,2.183460324257085  ,2.169217086996968  ,-0.419730389956033  ,0.399045961528183  ,-0.034268217013201  ,
# 1.510722178937970  ,-1.466779440533161  ,4.095498040373004  ,2.196629865500162  ,2.196080637537195  ,-0.415158468856053  ,0.394630184330989  ,-0.035317821117088  ,
# 1.503929729092387  ,-1.455452307207426  ,4.071990358580678  ,2.183507579330861  ,2.152332335109451  ,-0.423930425360864  ,0.399303924931203  ,-0.038627822160247  ,
# 1.500757512342094  ,-1.459867196391059  ,4.079183857076709  ,2.205017734822392  ,2.173320165977908  ,-0.425417448330034  ,0.403566492285593  ,-0.038899792407317  ,
# 1.503324820002883  ,-1.461215431985167  ,4.079364934100976  ,2.204191062769707  ,2.171838490023860  ,-0.446286992134591  ,0.403466621407424  ,-0.038916530348358  ,
# 1.503727618373531  ,-1.463985066457075  ,4.080761307102312  ,2.212237170452660  ,2.169276924378361  ,-0.427238077567021  ,0.411305562948692  ,-0.039070550802724  ,
       1.503853067655083  ,-1.463436057923992  ,4.080150880272005  ,2.214136548993840  ,2.171640890674952  ,-0.427141081840896  ,0.413509069674418  ,-0.039069258761021  ,         
                
                
                # He-He 
                #  1.61712964, -3.670e-01,  4.789e-01 ,-3.762e-01,  3.23425928, -2.760e-02,  4.344e-02, -7.470e-02, 
                 1.62963646, -0.3658247,   0.48055115, -0.36578079,  3.23371698, -0.02758561, 0.04354402, -0.07569902,

                # H-He
                # 0.999804742769  ,-0.217924286909  ,0.616280245192  ,-3.414516788600  ,1.850828728550  ,-0.098898299141  ,0.075722340000  ,-0.027001780000  ,
                # 1.00000043, -0.21809812,  0.61560816, -3.40915359,  1.8661813,  -0.09889802, 0.07572234, -0.02700178,
                # 0.749147949257  ,0.841331542711  ,0.665070223970  ,-0.683858613169  ,1.945668028291  ,-0.183712837285  ,0.186051744950  ,0.099854090129  ,
                # 1.052419908990098  ,-0.217924286909000  ,0.616280245192000  ,-3.414516788600000  ,1.850828728550000  ,-0.098898299141000  ,0.075722340000000  ,-0.027001780000000  ,
                # 1.112117319203040  ,-0.197105171459421  ,0.535217833548196  ,-3.093149326457217  ,2.171628534671961  ,-0.099093282122307  ,0.089361495131415  ,-0.026229530424303  ,
                # 1.389246640024114  ,-0.235774337690488  ,0.530729326182819  ,-3.021054330953968  ,1.862930853611370  ,-0.085354823325725  ,0.083669728367462  ,-0.023234843971784  ,
            #    1.061047378889416  ,-0.382019375230580  ,0.767663197400286  ,-3.158800337619714  ,1.901748209820538  ,-0.112794815986453  ,0.139810481406871  ,-0.194870568439244  ,
            #    1.06411536, -0.39520788,  0.79748191, -3.35673318,  1.847747 ,  -0.12264261, 0.15880473, -0.23018925,
# 1.062806371993989  ,-0.396125345216717  ,0.801932928188837  ,-3.396647583692781  ,1.845726226191536  ,-0.122792356884467  ,0.159264319354548  ,-0.232146198186701  ,
# 1.066163832524131  ,-0.397679890780384  ,0.802829066991810  ,-3.437749356314377  ,1.841785137383863  ,-0.123480865770139  ,0.160482164580854  ,-0.234114578438788  ,
                # 1.04626844, -0.33941331,  0.86443562, -3.82144771,  1.76209442, -0.11331669, 0.10277285, -0.08624397,
                # 1.046093681051500  ,-0.339691053356168  ,0.866413284274637  ,-3.844781302012729  ,1.761727892404334  ,-0.113318249454093  ,0.103018796090617  ,-0.088858503457306  ,
                # 1.07444089, -0.32392346 , 0.82255069, -3.88500038,  1.73938636, -0.11506669, 0.10185475, -0.08290284,
                # 1.042229043335329  ,-0.340770793334825  ,0.519314705977125  ,-2.927848634055453  ,2.035806391796232  ,-0.117687562350735  ,0.123935541628657  ,-0.092381078856146  ,
# 1.105702867895969,   -0.359930543324723,   0.533018253972319 ,  -3.003344785081300 ,  1.604496724417533,   -0.114455203526953 ,  0.106150760771522,   -0.104468490875411   ,
# 0.882902514315924  ,-0.350587247318660  ,0.552859489580492  ,-2.504192068936684  ,1.851958863771181  ,-0.121427737009826  ,0.123907451787507  ,-0.102999484139996  ,
# 1.046535631623660  ,-0.230138166123363  ,0.534108304032331  ,-3.497508489692005  ,1.912887990208797  ,-0.110073780412212  ,0.051378945913710  ,-0.015103421856352  ,
# 1.046535621155585  ,-0.230138195664522  ,0.512963504591793  ,-3.447849873685992  ,1.974440569902708  ,-0.104645336890045  ,0.051116362277277  ,-0.027601492744296  ,
# 1.023917920830613  ,-0.231248952209809  ,0.515522023821579  ,-3.568125651256524  ,1.979182229312970  ,-0.104844843244159  ,0.051676094082639  ,-0.027720129862740  ,
                1.029187717904903  ,-0.230617861727848  ,0.516418837384891  ,-3.563055960429090  ,1.976148194654602  ,-0.104694805988491  ,0.051599506311914  ,-0.027733679796375  ,
                
                # W-He p
                #  0.398127020000  ,-0.433621820000  ,0.055868330000  ,3.293364500000  ,-0.012057520000  ,-0.085603570000  ,0.102061370000
# 0.391719754353014  ,-0.449474505290761  ,0.057007212684928  ,3.573503468791126  ,-0.012120033028773  ,-0.063631614517639  ,0.112825301563015  ,
# 0.429341551300857  ,-0.454053797509829  ,0.060960603020105  ,3.405671323695183  ,-0.012390626523713  ,-0.060423150774040  ,0.110829181924681  ,
# 0.422398402272882  ,-0.460346236797038  ,0.063155340110237  ,3.485302422511698  ,-0.011767478281304  ,-0.060115842875619  ,0.112516229721144  ,
# 0.426836690998372  ,-0.456073392504887  ,0.061238555400597  ,3.442408895144805  ,-0.012445739040005  ,-0.060487965061169  ,0.110947884538627  ,
# 0.434668037213712  ,-0.458941066949467  ,0.062967042162744  ,3.467584421169657  ,-0.012577061010963  ,-0.059258366812841  ,0.109874915882536  ,
# 0.434872367789113  ,-0.458941051183554  ,0.062967937336788  ,3.467584774301574  ,-0.012577013687685  ,-0.059258486587022  ,0.109882216033569  ,
                #  0.45798955, -0.43436943, -0.04688532,  3.27458333, -0.01211176, -0.08546036, 0.10205941
# 0.435185130765343  ,-0.458377279388363  ,0.063266008538068  ,3.462933369105929  ,-0.012609431137626  ,-0.059165451289865  ,0.109545328096472  ,
# 0.435099532131648  ,-0.458233113986564  ,0.063228346712166  ,3.462937809497808  ,-0.012618707651871  ,-0.059167163526471  ,0.109565036916869  ,
# 0.435298736291955  ,-0.458471594515113  ,0.063847238234417  ,3.460871443095233  ,-0.012626800629147  ,-0.059277334290332  ,0.109394147726477  ,
0.437131642955087  ,-0.459843876423541  ,0.063966101563607  ,3.449811336911679  ,-0.012619206722510  ,-0.059462073676808  ,0.109857772586596  ,
])

print(x.shape, eam_fit.gen_rand().shape)
# x = np.array([   6.08600e-01,  2.78500e-01, 
#                  2.53408e+01,  6.75140e+00, 0,
#                  0, 0, 0,
#                  2.40898176e+00,  6.03993438e+00,  1.00000000e+00,
#                  0, 0, 0,
#                 -6.46700e-01,   1.13620e+00 ,-1.34460e+00, -3.12900e-01,  5.53600e-01,  4.46000e-02 ,  6.57300e-01,
#                 -3.670e-01,  4.789e-01 ,-3.762e-01, -2.760e-02,  4.344e-02, -7.470e-02, 
#                 -0.12564656, 0.13166891, -0.23713911, -0.02355287 , 0.02697471 ,-0.04887022,
#                  6.57300e-01, -4.56100e-01, -5.06000e-02,  2.86000e-02, -1.43000e-02,  -1.01000e-02])
eam_fit.sample_to_file(x)

print(x[eam_fit.map['H-He']])
Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, eam_fit.lammps_param['potfile'])

Handle_PotFiles_He.write_pot(eam_fit.pot_lammps, eam_fit.potlines, 'git_folder/Potentials/final.eam.he')

plt.plot(rho[:1000], eam_fit.pot_lammps['H F'][:1000])
plt.plot(rho[:1000], eam_fit.pot_lammps['He F'][:1000])
plt.plot(rho[:1000], eam_fit.pot_lammps['W F'][:1000])

plt.show()


r = np.linspace(0, eam_fit.pot_params['rc'], eam_fit.pot_params['Nr'])[1:]
plt.plot(data_dft[:,0], data_dft[:,1], label='dft', color='black')

pot = eam_fit.pot_lammps['H-He'][1:]/r
plt.plot(r[125:], pot[125:], label='pairwise component')

plt.scatter(h_he_ref[:,0], h_he_ref[:,1], label='qmc', color='red')
plt.show()