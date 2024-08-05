
''' Read and Write Potential Files '''

import numpy as np

def write_pot(pot, starting_lines, file_path):
    
    keys = [
            'W F' , 'W-W p' , 'W-H p' , 'W-He p' ,
            'H F' , 'H-W p' , 'H-H p' , 'H-He p' ,
            'He F', 'He-W p', 'He-H p', 'He-He p',
            'W-W' , 'W-H'   , 'H-H'   , 'W-He'  , 'H-He'   , 'He-He'
            ] 

    with open(file_path, 'w') as file:

        lines = starting_lines.split('\n')
        line_idx = 0

        for i in range(5):
            file.write(lines[line_idx] + '\n')
            line_idx += 1

        val = lines[4].split()

        Nrho = int(val[0])
        drho = float(val[1])
        Nr   = int(val[2])
        dr   = float(val[3])
        cutoff  = float(val[4])

        for element in ['W', 'H', 'He']:

            file.write(lines[line_idx] + '\n')

            line_idx += 1

            for i in range(Nrho):
                file.write('%16.8f \n' % pot[element + ' F'][i])
            
            for i in range(Nr):
                file.write('%16.8f \n' % pot[element + '-W p'][i])

            for i in range(Nr):
                file.write('%16.8f \n' % pot[element + '-H p'][i])

            for i in range(Nr):
                file.write('%16.8f \n' % pot[element + '-He p'][i])

        
        for pair in ['W-W', 'W-H', 'H-H', 'W-He', 'H-He', 'He-He']:
            
            for i in range(Nr):
                
                file.write('%16.8f \n' % pot[pair][i])

def read_pot(potfile_path):

    starting_lines = ''

    keys = [
            'W F' , 'W-W p' , 'W-H p' , 'W-He p' ,
            'H F' , 'H-W p' , 'H-H p' , 'H-He p' ,
            'He F', 'He-W p', 'He-H p', 'He-He p',
            'W-W' , 'W-H'   ,  'H-H' , 'W-He'  , 'H-He'   , 'He-He'
            ] 

    pot = {}

    for key in keys:
        pot[key] = []

    with open(potfile_path, 'r') as ref:
            
        for i in range(5):
            starting_lines += ref.readline()

        val = starting_lines.split('\n')[-2].split()

        Nrho = int(val[0])
        drho = float(val[1])
        Nr   = int(val[2])
        dr   = float(val[3])
        cutoff  = float(val[4])

        for element in ['W', 'H', 'He']:

            starting_lines += ref.readline()

            for i in range(Nrho):
                val = float(ref.readline())
                pot[element + ' F'].append(val)
                
            for i in range(Nr):
                val = float(ref.readline())
                pot[element + '-W p'].append(val)
        
            for i in range(Nr):
                val = float(ref.readline())
                pot[element + '-H p'].append(val)

            for i in range(Nr):
                val = float(ref.readline())
                pot[element + '-He p'].append(val)

        for pair in ['W-W', 'W-H', 'H-H', 'W-He', 'H-He', 'He-He']:
            
            for i in range(Nr):
                
                val = float(ref.readline())
                pot[pair].append(val)

    for key in pot:
        pot[key] = np.array(pot[key])

    return pot, starting_lines, {'Nrho': Nrho, 'drho':drho, 'Nr':Nr, 'dr':dr, 'rc':cutoff, 'rho_c':(Nrho-1)*drho}



