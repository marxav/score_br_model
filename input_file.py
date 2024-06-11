import os

def create_unique_input_file(input_file, verbose=False):
    
    l1_filename = input_file
    if l1_filename.endswith('_br.txt'):
        l2_filename = input_file[:-7] + '_fr.txt'
        br_file = l1_filename
    elif l1_filename.endswith('_fr.txt'):        
        l2_filename = input_file[:-7] + '_br.txt'
        br_file = l2_filename
    else:
        print(f'ERROR: input_file {input_file}: unexpected format')
        exit(-1)

    if not os.path.isfile(l1_filename):
        print(f'input file l1 {l1_filename} not found') 
        exit(-1)   
    if not os.path.isfile(l2_filename):
        print(f'input file l2 {l2_filename} not found') 
        exit(-1)

    # Open each of the two files in read mode and extract their text
    with open(l1_filename, 'r') as file:
        l1_text = file.read()
    with open(l2_filename, 'r') as file:
        l2_text = file.read()

    # split the src and dst texts over multiple lines (one sentence per line)
    l1_text = l1_text.rstrip().replace('.', '.\n')
    if l1_text[-1]=='\n':
        l1_text = l1_text[:-1]
    l1_lines = l1_text.split('\n')
    l2_text = l2_text.rstrip().replace('.', '.\n')
    if l2_text[-1]=='\n':
        l2_text = l2_text[:-1]
    l2_lines = l2_text.split('\n')

    if verbose:
        print('input_file:', input_file)
        print('l1_lines:', l1_lines)
        print('l2_lines:', l2_lines)
        print('len(l1_lines):', len(l1_lines))
        print('len(l2_lines):', len(l2_lines))

    # check that both files contains the same number of sentences
    assert len(l1_lines) == len(l2_lines), 'br and fr files do not have the same number of sentences'

    # write both text in a new tsv file
    new_input_file = input_file[:-7]+'.tsv'        
    with open(new_input_file, 'w+') as new_file:
        new_file.write('br'+'\t'+'fr'+'\n')
        if br_file == l1_filename:        
            for l1, l2 in zip (l1_lines, l2_lines):
                new_file.write(l1 + '\t' + l2 + '\n')
        else:
            for l2, l1 in zip (l2_lines, l1_lines):
                new_file.write(l2 + '\t' + l1 + '\n')
        new_file.close()
    
    return new_input_file
    
    
def check_input_file(args): 
    # minimum check of the input file extension
    if len(args) < 2:
        print('first argument requires an .tsv or .txt input file.')
        exit(-1)
    input_file = args[1]
    if not os.path.exists(input_file):
        print('input_file does not exist')
        exit(-1)
    elif input_file.endswith('.tsv'):
        return input_file
    elif input_file.endswith('_br.txt') or input_file.endswith('_fr.txt'):
        return create_unique_input_file(input_file)
    else:
        print('input_file type is not supported')
        exit(-1)