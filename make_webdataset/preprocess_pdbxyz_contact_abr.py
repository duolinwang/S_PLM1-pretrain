import argparse
import numpy as np
import os
import sys
import shutil
import yaml
from Bio.PDB import PDBParser
import glob


def encode_sequence(seq):
    alphabet = ["A", "C", "D", "E", "F", "G","H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    char_to_int = dict((c,i) for i,c in enumerate(alphabet))
    integer_encoded = [char_to_int[char] for char in seq]  # convert the protein sequence to the number sequence
    one_hot_encode = [] 
        
    for value in integer_encoded:
      letter = [0 for _ in range(len(alphabet))]
      letter[value] = 1
      one_hot_encode.append(letter) # each row only has one "1", represnting the type of residue
        
        #print(one_hot_encode)    
    return np.array((one_hot_encode)) # create a 2-D array for one_hot array.

def pad_seq(seq,max_len): #  padding
    leng=len(seq)
    if leng<max_len:
        pad_len = max_len-leng
        return np.pad(list(seq), (0, pad_len), mode='constant', constant_values=0)
    else:
        return np.asarray(list(seq[:max_len]))
    

def pad_coordinates(x,max_len): 
    leng = len(x)
    if leng>=max_len:
        return np.asarray(x[:max_len]) # we trim the contact map 2D matrix if it's dimension > max_len
    else:
       pad_len=max_len-leng
       return np.pad(x, [(0, pad_len),(0,0)], mode='constant', constant_values=0)

def truncate_seq(seq,max_len): #  padding
    return seq[:max_len]
    

def truncate_coordinates(x,max_len): 
     return np.asarray(x[:max_len])

def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), float)
    for row, residue_one in enumerate(chain_one) :
        for col, residue_two in enumerate(chain_two) :
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

def truncate_concatmap(matrix,max_len): 
    
    return matrix[:max_len,:max_len] # we trim the contact map 2D matrix if it's dimension > max_len

def extract_coordinates(chain) :
    """Returns a list of C-alpha coordinates for a chain"""
    pos=[]
    for row,residue in enumerate(chain):
        pos.append(residue["CA"].coord)
    
    return pos



parser = argparse.ArgumentParser(description='Pretreat')
parser.add_argument('-data', metavar='DIR', default='./pdbsamples3000/folder-4/', help='path to dataset')
parser.add_argument('-max_len', default=512, type=int, help='max sequence len to consider.')
parser.add_argument('-outputfolder', metavar='DIR', default='./pdbsamples3000/swiss-pos-contact/', help='path to output')
parser.add_argument('-index_base', default=1500, type=int, help='start index for this folder,should be ($j-1)*500')
args = parser.parse_args()

file_list = glob.glob(args.data + "*.pdb")
data_path = []
for path in os.listdir(args.data):
    if os.path.isfile(os.path.join(args.data, path)) and path[-3:] == 'pdb':
        data_path.append(os.path.join(args.data, path))

print(data_path)


for i in range(len(data_path)):
    #print(i)
    index=args.index_base+i
    print(index)
    outputfile=args.outputfolder+"/"+data_path[i].split("/")[-2]+"-"+str(i)+".npz"
    parser = PDBParser()# This is from Bio.PDB
    structure = parser.get_structure('protein', data_path[i])
    model = structure[0]
    sequence=[]
    chain=model['A']
    for residue_index,residue in enumerate(chain):
        sequence.append(residue.resname)
        dictn = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        amino_acid = [dictn[triple] for triple in sequence]
        protein_seq = "".join(amino_acid)
    
    ca_pos = extract_coordinates(chain)
    #pad_ca_pos=pad_coordinates(ca_pos,args.max_len)
    pad_ca_pos=truncate_coordinates(ca_pos,args.max_len)
    #pad_seq=pad_seq(protein_seq,args.max_len)
    pad_seq=truncate_seq(protein_seq,args.max_len)
    contactmap = truncate_concatmap(calc_dist_matrix(chain,chain),args.max_len)
    np.savez(outputfile,index=index,seq=pad_seq,ca_pos=pad_ca_pos,contactmap=contactmap)
