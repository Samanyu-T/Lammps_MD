import sys

def edit_data_files(filepath):

    with open(filepath, 'r') as file:
        lines = file.readlines()
        
    lines[3] = '           3  atom types \n'
    lines.insert(12, '            2   1.00784000            # H \n')
    lines.insert(13, '            3   4.00264000            # He \n')

    with open(filepath, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    edit_data_files(sys.argv[1])
