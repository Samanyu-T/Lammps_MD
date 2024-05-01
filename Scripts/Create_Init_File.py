
import json

param = {}

param['alattice']  = 3.144221

param['orientx'] =  [1, 0, 0]
param['orienty'] =  [0, 1, 0]
param['orientz'] =  [0, 0, 1]

param['size'] =  7

param['surface'] =  20

param['potfile'] = '../Potentials/WHHe_test.eam.alloy'

param['conv'] =  1000

param['machine'] = ''

with open('init_param.json', "w") as json_file:
    json.dump(param, json_file, indent=4)
