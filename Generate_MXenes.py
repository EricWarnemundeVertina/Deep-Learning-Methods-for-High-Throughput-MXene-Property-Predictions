print("Hello, world!")

import sys
import os
import argparse
import numpy as np
from pymatgen.core import Structure
import pandas as pd
import math
import time
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter
import random
from collections import Counter
import itertools
import re




parser = argparse.ArgumentParser(description="Get MXene structures")

parser.add_argument(
    "--configuration_type", default="", help="Structure stacking type and termination location"
)

parser.add_argument(
    "--n_value", default="", help="Size of n, such as n7"
)

parser.add_argument(
    "--M_combination", default="", help="Number of M groups in these MXenes; e.g., 3-2-2-1"
)

parser.add_argument(
    "--save_path", default="", help="Directory where .cif files will be saved"
)




M_dict = {'M1':['Sc', 'Y'],
       'M2': ['Ti', 'Zr', 'Hf'],
       'M3':['V', 'Nb', 'Ta'],
       'M4':['Cr', 'Mo', 'W'],
       'M5':['Mn', 'Tc', 'Re'],
       'M6':['Fe', 'Co', 'Ni', 'Cu'],
       'M7':['Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt', 'Ag', 'Au'],
       'M8':['Zn', 'Cd', 'Hg'],
       'M9':['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
       'M10':['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu'],
       'M11':['Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Po'],
       'M12':['B', 'Si', 'Ge', 'As', 'Sb', 'Bi', 'At']}

X_dict = {'X1':['C'],
       'X2':['N'],
       'X3':['B'],
       'X4':['Al', 'Ga', 'Sn', 'Si', 'Ge'],
       'X5':['P', 'As', 'Sb', 'Bi'],
       'X6':['O', 'S', 'Se', 'Te']}


#Tx_dict = {'Tx2':['O', 'S', 'Se', 'Te'],
#       'Tx3':['F', 'Cl', 'Br', 'I'],
#       'Tx4':['H', 'P', 'As', 'N', 'Sb'],
#       'Tx5':[['F', 'O'], ['F', 'S'], ['F', 'Se'], ['F', 'Te'], ['Cl', 'O'], ['Cl', 'S'], ['Cl', 'Se'], ['Cl', 'Te'], ['Br', 'O'], ['Br', 'S'], ['Br', 'Se'], ['Br', 'Te'],
#               ['I', 'O'], ['I', 'S'], ['I', 'Se'], ['I', 'Te']],
#       'Tx6':[['H', 'F'], ['H', 'Cl'], ['H', 'Br'], ['H', 'I']], 
#       'Tx7':[['H', 'O'], ['H', 'S'], ['H', 'Se'], ['H', 'Te'], ['H', 'N']]}  # Halogen + Chalcogen, H + Chalcogen,  # H + Halogen, H-N, 

Tx_dict = {'Tx6':[['H', 'F'], ['H', 'Cl'], ['H', 'Br'], ['H', 'I']]}  # Halogen + Chalcogen, H + Chalcogen,  # H + Halogen, H-N, 

structure_types = ['ABC', 'ABChM', 'ABChX', 'ABCm', 'ABCt', 'ABA', 'ABAh', 'ABAhX', 'ABAm', 'ABAt']



def sum_integers_in_string(s):  # add all digits in a string; this is a count of the number of atoms in a string
    numbers = re.findall(r'\d+', s)  # Find all sequences of digits
    return sum(int(num) for num in numbers)


def repeat_segments_by_integers(s):  # given a string of a chemical formula, return a list where each entry is a the element symbol
    # each element symbol is repeated n times, where n is the number of atoms of that element in the string
    parts = re.split(r'\d+', s)
    numbers = list(map(int, re.findall(r'\d+', s)))

    result = []
    for i in range(len(numbers)):
        result.extend([parts[i]] * numbers[i])
    
    return result



def rotate_z(x, y, z, angle_deg):
    theta = math.radians(angle_deg)
    x_rot = x * math.cos(theta) - y * math.sin(theta)
    y_rot = x * math.sin(theta) + y * math.cos(theta)
    return x_rot, y_rot, z  # z stays the same


def get_x(lattice_a, x_frac, y_frac):
    return lattice_a*(x_frac - y_frac/2)


def get_y(lattice_a, y_frac):
    return math.sqrt(3)*lattice_a*y_frac/2


def save_generated_MXene(save_path, save_filename, lattice_a, lattice_c, atom_list, frac_coords):
    lattice = Lattice.hexagonal(lattice_a, lattice_c)
    structure = Structure(lattice, atom_list, frac_coords)
    writer = CifWriter(structure)
    writer.write_file(save_path + save_filename)

    return 'Done!'


def generate_MXene(configuration_type, num_MX_atoms, avg_M_X_bonds_df, avg_M_Tx_bonds_df, avg_Tx_Tx_bonds_df, x_frac_list, y_frac_list, Tx_x_frac_list, Tx_y_frac_list, theta_MX_list, theta_MTx_list, phi_MX_list, phi_MTx_list, atom_list, desired_layer_distance, num_Tx, Tx_atoms_list):

    min_bond_len = 4
    longest_bond_pair_indexes = []

    for i in range(len(atom_list) - 1):
        if num_Tx == 2 and (i == 0 or i == len(atom_list) - 2):  # do not want to base the cell on the Tx-Tx bond length
            if Tx_atoms_list[0] == 'H' and Tx_atoms_list[1] in ['F','Cl','Br','I']:
                new_bond = avg_Tx_Tx_bonds_df.loc[Tx_atoms_list[0], Tx_atoms_list[1]]
            else:
                new_bond = avg_Tx_Tx_bonds_df.loc[Tx_atoms_list[1], Tx_atoms_list[0]]
        elif (num_Tx == 2 and i == 1) or (num_Tx == 1 and i == 0):  # bottom M-Tx bond
            new_bond = avg_M_Tx_bonds_df.loc[atom_list[i + 1], atom_list[i]]
            if not ('t' in configuration_type) and new_bond < min_bond_len:
                min_bond_len = new_bond
                longest_bond_pair_indexes = i
        elif (num_Tx == 2 and i == len(atom_list) - 3) or (num_Tx == 1 and i == len(atom_list) - 2):  # top M-Tx bond
            new_bond = avg_M_Tx_bonds_df.loc[atom_list[i], atom_list[i + 1]]
            if not ('t' in configuration_type) and new_bond < min_bond_len:
                min_bond_len = new_bond
                longest_bond_pair_indexes = i
        elif (i-num_Tx)%2 == 0: # M element
            new_bond = avg_M_X_bonds_df.loc[atom_list[i], atom_list[i + 1]]
            if new_bond < min_bond_len:
                min_bond_len = new_bond
                longest_bond_pair_indexes = i
        else: # X element
            new_bond = avg_M_X_bonds_df.loc[atom_list[i + 1], atom_list[i]]
            if new_bond < min_bond_len:
                min_bond_len = new_bond
                longest_bond_pair_indexes = i

    if 'ABA' in configuration_type:
        if (longest_bond_pair_indexes > num_Tx - 1) and (longest_bond_pair_indexes < len(atom_list) - num_Tx - 1): # M-X bond
            if longest_bond_pair_indexes == num_Tx or longest_bond_pair_indexes == len(atom_list) - num_Tx - 2:  # bottommost or toppmost M-X bond atom
                theta = theta_MX_list[0]
            else:  # not bottommost or toppmost M atom
                theta = theta_MX_list[1]
            phi = phi_MX_list[longest_bond_pair_indexes%2]
        elif longest_bond_pair_indexes == num_Tx - 1: # bottom M-Tx bond
            theta = theta_MTx_list[0]
            phi = phi_MTx_list[0]
        elif longest_bond_pair_indexes == len(atom_list) - num_Tx - 1: # top M-Tx bond
            theta = theta_MTx_list[1]
            phi = phi_MTx_list[1]
    else:  # ABC
        if (longest_bond_pair_indexes > num_Tx - 1) and (longest_bond_pair_indexes < len(atom_list) - num_Tx - 1): # M-X bond
            if longest_bond_pair_indexes == num_Tx or longest_bond_pair_indexes == len(atom_list) - num_Tx - 2:  # bottommost or toppmost M-X bond
                theta = theta_MX_list[0]
            else:  # not bottommost or toppmost M atom
                theta = theta_MX_list[1]
            phi = phi_MX_list[longest_bond_pair_indexes%3]
        elif longest_bond_pair_indexes == num_Tx - 1: # bottom M-Tx bond
            theta = theta_MTx_list[0]
            phi = phi_MTx_list[0]
        elif longest_bond_pair_indexes == len(atom_list) - num_Tx - 1: # top M-Tx bond
            theta = theta_MTx_list[1]
            phi = phi_MTx_list[1]
        else:
            print('Problem with indexing atom list for theta, phi!!!')
        # don't need angles between Tx-Tx bonds, since entire bond is in z-direction


    r = min_bond_len
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    x2, y2, z2 = rotate_z(x, y, z, 120)
    lattice_a = math.sqrt((x2-x)**2 + (y2-y)**2)
    x_frac_list_generated = []
    y_frac_list_generated = []
    z_list = []
    for i in range(len(atom_list) - 1):  
        if i < num_Tx:  # bottom Tx
            if i == 0:  # bottommost Tx atom
                xprev_frac, yprev_frac = Tx_x_frac_list[0], Tx_y_frac_list[0]
                xprev, yprev = get_x(lattice_a, xprev_frac, yprev_frac), get_y(lattice_a, yprev_frac)
                znext, zprev = 0, 0
                x_frac_list_generated.append(xprev_frac)
                y_frac_list_generated.append(yprev_frac)
                z_list.append(zprev + desired_layer_distance/2)
                if num_Tx == 2:
                    if atom_list[i] == 'H' and atom_list[i + 1] in ['F','Cl','Br','I']:
                        new_bond = avg_Tx_Tx_bonds_df.loc[atom_list[i], atom_list[i + 1]]
                    else:
                        new_bond = avg_Tx_Tx_bonds_df.loc[atom_list[i + 1], atom_list[i]]
                    xnext_frac, ynext_frac = xprev_frac, yprev_frac
                    xnext, ynext = xprev, yprev  # outermost Tx atom in Tx-Tx bond has same x and y coordinates as other Tx atom
                else:  # num_Tx is 1    # M-Tx bond
                    new_bond = avg_M_Tx_bonds_df.loc[atom_list[i + 1], atom_list[i]]
                    xnext_frac, ynext_frac = x_frac_list[0], y_frac_list[0]
                    xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)
            elif i == 1:  # 2nd Tx-Tx atom
                new_bond = avg_M_Tx_bonds_df.loc[atom_list[i + 1], atom_list[i]]
                xnext_frac, ynext_frac = x_frac_list[0], y_frac_list[0]
                xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)
        elif num_Tx == 2 and i == len(atom_list) - num_Tx:   # bottommost Tx atoms in top Tx-Tx bond
            if atom_list[i + 1] == 'H' and atom_list[i] in ['F','Cl','Br','I']:
                new_bond = avg_Tx_Tx_bonds_df.loc[atom_list[i + 1], atom_list[i]]
            else:
                new_bond = avg_Tx_Tx_bonds_df.loc[atom_list[i], atom_list[i + 1]]
            xnext_frac, ynext_frac = xprev_frac, yprev_frac
            xnext, ynext = xprev, yprev
        else:  # M atom or X atom
            if i == 0:  # bare MXene
                new_bond = avg_M_X_bonds_df.loc[atom_list[i], atom_list[i + 1]]
                xprev_frac, yprev_frac = x_frac_list[0], y_frac_list[0]
                xprev, yprev = get_x(lattice_a, xprev_frac, yprev_frac), get_y(lattice_a, yprev_frac)
                xnext_frac, ynext_frac = x_frac_list[1], y_frac_list[1]
                xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)
                znext, zprev = 0, 0
                x_frac_list_generated.append(xprev_frac)
                y_frac_list_generated.append(yprev_frac)
                z_list.append(zprev + desired_layer_distance/2)
            elif i == num_MX_atoms + num_Tx - 1:  # top M-Tx bond
                new_bond = avg_M_Tx_bonds_df.loc[atom_list[i], atom_list[i + 1]]
                xnext_frac, ynext_frac = Tx_x_frac_list[1], Tx_y_frac_list[1]
                xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)
            elif (i-num_Tx)%2 == 0:  # M atom
                new_bond = avg_M_X_bonds_df.loc[atom_list[i], atom_list[i + 1]]
                if 'ABA' in configuration_type:  # ABA
                    xnext_frac, ynext_frac = x_frac_list[1], y_frac_list[1]
                else:  # ABC
                    xnext_frac, ynext_frac = x_frac_list[(i-num_Tx+1)%3], y_frac_list[(i-num_Tx+1)%3]
                xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)
            else: # X atom
                new_bond = avg_M_X_bonds_df.loc[atom_list[i + 1], atom_list[i]]
                if 'ABA' in configuration_type:  # ABA
                    xnext_frac, ynext_frac = x_frac_list[0], y_frac_list[0]
                else:  # ABC
                    xnext_frac, ynext_frac = x_frac_list[(i-num_Tx+1)%3], y_frac_list[(i-num_Tx+1)%3]
                xnext, ynext = get_x(lattice_a, xnext_frac, ynext_frac), get_y(lattice_a, ynext_frac)


        x_frac_list_generated.append(xnext_frac)
        y_frac_list_generated.append(ynext_frac)
        znext = zprev + math.sqrt(new_bond**2 - (xnext - xprev)**2 - (ynext-yprev)**2)
        xprev_frac, yprev_frac = xnext_frac, ynext_frac
        xprev, yprev = xnext, ynext
        zprev = znext
        z_list.append(znext + desired_layer_distance/2)

    lattice_c = znext + desired_layer_distance

    z_frac_list = np.array(z_list) / lattice_c
    frac_coords = [list(frac_coord) for frac_coord in zip(x_frac_list_generated, y_frac_list_generated, z_frac_list)]

    return lattice_a, lattice_c, frac_coords





def new_MXene_atoms(configuration_type, n_value, M_combination, selected_M_groups, selected_X_group, selected_Tx_group):
    
    num_each_M_list = [int(x) for x in M_combination.split("-")]

    M_atoms_list = []
    for i in range(len(num_each_M_list)):
        M_to_append = []
        M_to_append = [random.choice(M_dict[selected_M_groups[i]])]*num_each_M_list[i]
        M_atoms_list += M_to_append

    X_atoms_list = random.choice(X_dict[selected_X_group])

    Tx_atoms_list = [random.choice(Tx_dict[selected_Tx_group])]

    if configuration_type == 'ABC' or configuration_type == 'ABA':
        num_Tx = 0
    else:
        if len(Tx_atoms_list[0]) == 2 and isinstance(Tx_atoms_list[0], list):
            num_Tx = 2
            Tx_atoms_list = Tx_atoms_list[0]
        elif 'none' in Tx_atoms_list:  # bare MXene
            num_Tx = 0
        else:  # single Tx on top and bottom
            num_Tx = 1
    if configuration_type == 'ABC' or configuration_type == 'ABA':
        num_atoms = 2*int(n_value[-1]) + 1
    else:
        num_atoms = 2*int(n_value[-1]) + 1 + 2*num_Tx

    return num_atoms, M_atoms_list, X_atoms_list, Tx_atoms_list, num_Tx





def get_avg_bond_dfs(directory, avg_M_X_bonds_name, avg_M_Tx_bonds_name, avg_Tx_Tx_bonds_name, n_value):

    if n_value in ['n5', 'n6', 'n7', 'n8', 'n9']:
        avg_M_X_bonds_filename = avg_M_X_bonds_name + '_n4' + '.csv'
        avg_M_Tx_bonds_filename = avg_M_Tx_bonds_name + '_n4' + '.csv'
        avg_Tx_Tx_bonds_filename = avg_Tx_Tx_bonds_name + '_n4' + '.csv'
    else:
        avg_M_X_bonds_filename = avg_M_X_bonds_name  + '_' + n_value + '.csv'
        avg_M_Tx_bonds_filename = avg_M_Tx_bonds_name + '_' + n_value + '.csv'
        avg_Tx_Tx_bonds_filename = avg_Tx_Tx_bonds_name + '_' + n_value + '.csv'

    avg_M_X_bonds_df = pd.read_csv(directory + avg_M_X_bonds_filename)
    column_names = avg_M_X_bonds_df.columns.tolist()
    avg_M_X_bonds_df = avg_M_X_bonds_df.set_index(column_names[0])

    avg_M_Tx_bonds_df = pd.read_csv(directory + avg_M_Tx_bonds_filename)
    column_names = avg_M_Tx_bonds_df.columns.tolist()
    avg_M_Tx_bonds_df = avg_M_Tx_bonds_df.set_index(column_names[0])

    avg_Tx_Tx_bonds_df = pd.read_csv(directory + avg_Tx_Tx_bonds_filename)
    column_names = avg_Tx_Tx_bonds_df.columns.tolist()
    avg_Tx_Tx_bonds_df = avg_Tx_Tx_bonds_df.set_index(column_names[0])

    return avg_M_X_bonds_df, avg_M_Tx_bonds_df, avg_Tx_Tx_bonds_df




def get_fracs_phis_thetas(configuration_type, n_value):
    num_MX_atoms = 2*int(n_value[-1]) + 1
    num_X_atoms = int(n_value[-1])
    if 'ABC' in configuration_type:
        x_frac_list = [0.666666666, 0.333333333, 0]  # periodic every 3 atoms (not including Tx atoms)
        y_frac_list = [0.333333333, 0.666666666, 0]  # periodic every 3 atoms (not including Tx atoms)
        theta_MX_list = [1.0383, 0.96747778]   # first value is for outermost M-X bonds, second value value is for all other M-X bonds (not toppmost or bottommost M-X bond)
        phi_MX_list = [2.6179938780, -1.5707963268, 0.5235987756]  #in radians; 150, -90, 30 in degrees; periodic every 2 atoms (not including Tx atoms)

        if 'hX' in configuration_type:  # ABC-hX
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[1], x_frac_list[(num_MX_atoms - 2)%3]], [y_frac_list[1], y_frac_list[(num_MX_atoms - 2)%3]] 
            theta_MTx_list = [0.709745072, 0.709839028]  # first value is for bottommost M-Tx bond, second value is for toppmost M-Tx bond
            phi_MTx_list =  [-0.523611171, 1.570785766]  # first value is for bottommost M-Tx bond, second value is for toppmost M-Tx bond
        elif 'hM' in configuration_type:  # ABC-hM
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[2], x_frac_list[(num_MX_atoms)%3]], [y_frac_list[2], y_frac_list[(num_MX_atoms)%3]]
            theta_MTx_list = [1.096549853, 1.09643109]
            phi_MTx_list =  [-1.5708304, 2.617994783]
        elif 'm' in configuration_type:  # ABC-m
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[1], x_frac_list[(num_MX_atoms)%3]], [y_frac_list[1], y_frac_list[(num_MX_atoms)%3]]
            theta_MTx_list = [0.805602738, 0.805667734]
            phi_MTx_list =  [-2.617992704, 0.523599949]
        elif 't' in configuration_type:  # ABC-t
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[0], x_frac_list[(num_MX_atoms - 1)%3]], [y_frac_list[0], y_frac_list[(num_MX_atoms - 1)%3]]
            theta_MTx_list = []  # do not want to base unit cells on M-Tx -t bonds
            phi_MTx_list =  []  # do not want to base unit cells on M-Tx -t bonds              
        else:  # ABC
            Tx_x_frac_list, Tx_y_frac_list = [], []   # there are not surface terminations
            theta_MTx_list = []   # there are not surface terminations
            phi_MTx_list =  []   # there are not surface terminations

    elif 'ABA' in configuration_type:
        x_frac_list = [0, 0.666666666]  # M atoms take 0, X atoms take 0.66; periodic every 2 atoms (not including Tx atoms)
        y_frac_list = [0, 0.333333333]  # M atoms take 0, X atoms take 0.33; periodic every 2 atoms (not including Tx atoms)
        theta_MX_list = [0.933751150, 0.895353906]  # periodic every 2 atoms (not including Tx atoms)
        phi_MX_list = [0.523598776, -2.617993878]   # 30 degrees and 150 degrees; periodic every 2 atoms (not including Tx atoms)

        if 'hX' in configuration_type:  # ABA-hX
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[1], x_frac_list[1]], [y_frac_list[1], y_frac_list[1]]
            theta_MTx_list = [0.943218912, 0.94319161] # first value is for bottommost M-Tx bond, second value is for toppmost M-Tx bond
            phi_MTx_list =  [-2.617992867, 0.523599786] # first value is for bottommost M-Tx bond, second value is for toppmost M-Tx bond
        elif 'h' in configuration_type:  # ABA-h
            Tx_x_frac_list, Tx_y_frac_list = [0.333333333, 0.333333333], [0.666666666, 0.666666666]
            theta_MTx_list = [0.700229196, 0.700187579]
            phi_MTx_list =  [2.617908274, -0.523683604]
        elif 'm' in configuration_type:  # ABA-m
            Tx_x_frac_list, Tx_y_frac_list = [x_frac_list[1], 0.333333333], [y_frac_list[1], 0.666666666]
            theta_MTx_list = [0.98068548, 0.980776007]
            phi_MTx_list =  [-2.617996765, 1.570796327]
        elif 't' in configuration_type:  # ABA-t
            Tx_x_frac_list, Tx_y_frac_list = [0, 0], [0, 0]
            theta_MTx_list = []  # do not want to base unit cells on M-Tx -t bonds
            phi_MTx_list =  []  # do not want to base unit cells on M-Tx -t bonds
        else:  # ABA
            Tx_x_frac_list, Tx_y_frac_list = [], []   # there are not surface terminations
            theta_MTx_list = []   # there are not surface terminations
            phi_MTx_list =  []   # there are not surface terminations
    else:
        print('Problem with configuration type!! Neither ABA nor ABC!')


    return num_MX_atoms, num_X_atoms, x_frac_list, y_frac_list, Tx_x_frac_list, Tx_y_frac_list, theta_MX_list, theta_MTx_list, phi_MX_list, phi_MTx_list

    

def get_combos(M_dict, X_dict, Tx_dict, M_combination):

    M_possibilities_list = list(M_dict.keys())
    num_unique_M_groups = len([int(x) for x in M_combination.split("-")])   # number of M groups for MXene to predict

    M_combinations_list = []
    if num_unique_M_groups == 1:
        M_combinations_list = [[m_group] for m_group in M_possibilities_list]
    else:
        M_combinations_list += [list(combo) for combo in itertools.combinations(M_possibilities_list, num_unique_M_groups)]

    X_possibilities_list = list(X_dict.keys())
    Tx_possibilities_list = list(Tx_dict.keys())

    return M_combinations_list, X_possibilities_list, Tx_possibilities_list





args = parser.parse_args(sys.argv[1:])
configuration_type = args.configuration_type # read this in with argparse
n_value = args.n_value # read this in with argparse -> can use job array numbers for this, perhaps
M_combination = args.M_combination   # read this in with arg parse
save_path = args.save_path


bond_directory = 'Average_bond_lengths/'  # directory to avg bond .csv's

avg_M_X_bonds_name = 'avg_M_X_bonds'
avg_M_Tx_bonds_name = 'avg_M_Tx_bonds'
avg_Tx_Tx_bonds_name = 'avg_Tx_Tx_bonds'

desired_layer_distance = 15



avg_M_X_bonds_df, avg_M_Tx_bonds_df, avg_Tx_Tx_bonds_df = get_avg_bond_dfs(bond_directory, avg_M_X_bonds_name, avg_M_Tx_bonds_name, avg_Tx_Tx_bonds_name, n_value)
num_MX_atoms, num_X_atoms, x_frac_list, y_frac_list, Tx_x_frac_list, Tx_y_frac_list, theta_MX_list, theta_MTx_list, phi_MX_list, phi_MTx_list = get_fracs_phis_thetas(configuration_type, n_value)
M_combinations_list, X_possibilities_list, Tx_possibilities_list = get_combos(M_dict, X_dict, Tx_dict, M_combination)



start = time.perf_counter()


# Generate new MXenes with some randomness
for selected_M_groups in M_combinations_list:
    for selected_X_group in X_possibilities_list:
        if configuration_type == 'ABC' or configuration_type == 'ABA':
            atom_list = []
            num_atoms, M_atoms_list, X_atoms_list, Tx_atoms_list, num_Tx = [], [], [], [], []
            selected_Tx_group = 'Tx2'
            num_atoms, M_atoms_list, X_atoms_list, Tx_atoms_list, num_Tx = new_MXene_atoms(configuration_type, n_value, M_combination, selected_M_groups, selected_X_group, selected_Tx_group)
            Tx_atoms_list = []
            num_Tx = 0
            X_atoms_list_name = [X_atoms_list]*num_X_atoms
            countsM = Counter(M_atoms_list)
            countsX = Counter(X_atoms_list_name)
            if not (configuration_type == 'ABC' or configuration_type == 'ABA'):
                countsTx = Counter(Tx_atoms_list*2)
                save_filename = ''.join(f'{elem}{countsM[elem]}' for elem in countsM) + ''.join(f'{elem}{countsX[elem]}' for elem in countsX) + ''.join(f'{elem}{countsTx[elem]}' for elem in countsTx) + '.cif'
            else:
                save_filename = ''.join(f'{elem}{countsM[elem]}' for elem in countsM) + ''.join(f'{elem}{countsX[elem]}' for elem in countsX) + '.cif'

            for j in range(num_atoms):
                if j%2 == 0:  # M atom
                    atom_list.append(M_atoms_list.pop(random.randrange(len(M_atoms_list))))
                else:  # X atom
                    atom_list.append(X_atoms_list)
            
            lattice_a, lattice_c, frac_coords = generate_MXene(configuration_type, num_MX_atoms, avg_M_X_bonds_df, avg_M_Tx_bonds_df, avg_Tx_Tx_bonds_df, x_frac_list, y_frac_list, Tx_x_frac_list, Tx_y_frac_list, theta_MX_list, theta_MTx_list, phi_MX_list, phi_MTx_list, atom_list, desired_layer_distance, num_Tx, Tx_atoms_list)

            save_generated_MXene(save_path + '/', save_filename, lattice_a, lattice_c, atom_list, frac_coords)
            #save_generated_MXene(fake_save_path, save_filename, lattice_a, lattice_c, atom_list, frac_coords)
        else:
            for selected_Tx_group in Tx_possibilities_list:
                atom_list = []
                num_atoms, M_atoms_list, X_atoms_list, Tx_atoms_list, num_Tx = [], [], [], [], []
                num_atoms, M_atoms_list, X_atoms_list, Tx_atoms_list, num_Tx = new_MXene_atoms(configuration_type, n_value, M_combination, selected_M_groups, selected_X_group, selected_Tx_group)

                X_atoms_list_name = [X_atoms_list]*num_X_atoms
                countsM = Counter(M_atoms_list)
                countsX = Counter(X_atoms_list_name)
                if not (configuration_type == 'ABC' or configuration_type == 'ABA'):
                    countsTx = Counter(Tx_atoms_list*2)
                    save_filename = ''.join(f'{elem}{countsM[elem]}' for elem in countsM) + ''.join(f'{elem}{countsX[elem]}' for elem in countsX) + ''.join(f'{elem}{countsTx[elem]}' for elem in countsTx) + '.cif'
                else:
                    save_filename = ''.join(f'{elem}{countsM[elem]}' for elem in countsM) + ''.join(f'{elem}{countsX[elem]}' for elem in countsX) + '.cif'

                for j in range(num_atoms):
                    if num_Tx == 2 and (j == 0 or j == num_atoms - 1):  # only need to append entire Tx list once on top, and once on bottom
                        pass
                    elif ((j == 0 and num_Tx == 1) or (j == 1 and num_Tx == 2)) and not (configuration_type == 'ABC' or configuration_type == 'ABA'):  # need to include surface terminations
                        atom_list += Tx_atoms_list  # reverse the order of the Tx list
                    elif ((j == num_atoms - 1 and num_Tx == 1) or (j == num_atoms - 2 and num_Tx == 2)) and not (configuration_type == 'ABC' or configuration_type == 'ABA'):
                        atom_list += Tx_atoms_list[::-1]
                    elif (j-num_Tx)%2 == 0:  # M atom
                        atom_list.append(M_atoms_list.pop(random.randrange(len(M_atoms_list))))
                    else:  # X atom
                        atom_list.append(X_atoms_list)
                
                lattice_a, lattice_c, frac_coords = generate_MXene(configuration_type, num_MX_atoms, avg_M_X_bonds_df, avg_M_Tx_bonds_df, avg_Tx_Tx_bonds_df, x_frac_list, y_frac_list, Tx_x_frac_list, Tx_y_frac_list, theta_MX_list, theta_MTx_list, phi_MX_list, phi_MTx_list, atom_list, desired_layer_distance, num_Tx, Tx_atoms_list)

                save_generated_MXene(save_path + '/', save_filename, lattice_a, lattice_c, atom_list, frac_coords)
                #save_generated_MXene(fake_save_path, save_filename, lattice_a, lattice_c, atom_list, frac_coords)





end = time.perf_counter()
print(f"Runtime: {end - start:.4f} seconds")




