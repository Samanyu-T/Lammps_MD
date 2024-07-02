import sys

def edit_data_files(filepath):

    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    splitx = [_str for _str in lines[5].split(' ') if _str != '']
    xlo = float(splitx[0])
    xhi = float(splitx[1]) + 300

    splity = [_str for _str in lines[6].split(' ') if _str != '']
    ylo = float(splity[0])
    yhi = float(splitx[1]) + 300

    lines[5] = '\t{:14.6g}\t{:14.6g}  xlo xhi\n'.format(xlo, xhi)
    lines[6] = '\t{:14.6g}\t{:14.6g}  ylo yhi\n'.format(ylo, yhi)

    lines[3] = '           3  atom types \n'
    lines.insert(12, '            2   1.00784000            # H \n')
    lines.insert(13, '            3   4.00264000            # He \n')

    with open(filepath, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    edit_data_files(sys.argv[1])
