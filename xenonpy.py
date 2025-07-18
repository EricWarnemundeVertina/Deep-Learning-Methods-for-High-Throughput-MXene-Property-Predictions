print("Hello, world!")

import re
from collections import defaultdict
import sys

import argparse
import pandas as pd
import numpy as np
from xenonpy.descriptor import Compositions
import ast

parser = argparse.ArgumentParser(description="XenonPy")

parser.add_argument("--target_prop", default="", help="target_prop")
parser.add_argument("--best_model", default="", help="best_model")
parser.add_argument("--data_filename", default="", help="data_filename")
parser.add_argument("--path", default="", help="path")
parser.add_argument("--save_filename", default="", help="save_filename")




def parse_formula(formula):
    """
    Parse a chemical formula into a dictionary of element counts.
    
    Parameters:
        formula (str): A chemical formula, e.g., 'Fe2O3'
        
    Returns:
        dict: Dictionary with element symbols as keys and counts as values
    """
    # Match elements and their counts: e.g., [('Fe', '2'), ('O', '3')]
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)

    composition = defaultdict(int)
    for element, count in matches:
        composition[element] += int(count) if count else 1
    
    return dict(composition)



def count_decimals(x):
    if pd.isnull(x):
        return 0
    s = str(x).split('.')
    return len(s[1]) if len(s) > 1 else 0



def round_df_cols(df, max_decimals, round_to):
    # Process each column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Count decimal places for each value
            decimal_counts = df[col].apply(count_decimals)
            if (decimal_counts > max_decimals).any():
                df[col] = df[col].round(round_to)
    
    return df




args = parser.parse_args(sys.argv[1:])
target_prop = args.target_prop # read this in with argparse
best_model = args.best_model # read this in with argparse
data_filename = args.data_filename # read this in with argparse
path = args.path # read this in with argparse
save_filename = args.save_filename # read this in with argparse



#get dictionary of material chemical formula to use for XenonPy
original_data_df = pd.read_csv(path + data_filename, header=None, names=['Name', 'Target'])
print('original_data_df: ', original_data_df)

original_data_df["xenonpy_dict_name"] = original_data_df["Name"].apply(parse_formula)
print('original_data_df with dictionary for XenonPy: ', original_data_df)
#original_data_df.to_csv(path + save_filename, index=False)





'''
#XenonPy transformation
original_data_df = pd.read_csv(path + data_filename)
original_data_df['xenonpy_dict_name'] = original_data_df['xenonpy_dict_name'].apply(ast.literal_eval)
print('original_data_df: ', original_data_df)
'''


materials_list = original_data_df['xenonpy_dict_name'].tolist()
comps = [[material] for material in materials_list]
print('comps: ', comps)



cal = Compositions()
#print('comps: ', comps)
descriptor = cal.transform(comps)
print('descriptor before targets: ', descriptor)


'''
cal = Compositions()
comps_list = []
for i in range(len(original_data_df)):
    #cal = Compositions()
    row_df = original_data_df.iloc[[i]]
    comps = row_df['xenonpy_dict_name'].tolist()
    print('comps: ', comps)

    descriptor_one_material = cal.transform(comps)
    print(descriptor_one_material)
    comps_list.append(descriptor_one_material)


descriptor = pd.concat(comps_list, ignore_index=True)
print('descriptor: ', descriptor)
'''


max_decimals = 4     # max allowed decimal digits before rounding
round_to = 4 

descriptor = round_df_cols(descriptor, max_decimals, round_to)
print("descriptor after rounding: ", descriptor)


#descriptor['Name'] = original_data_df['Name']
descriptor.insert(0, 'Name', original_data_df['Name'])
descriptor['Target'] = original_data_df['Target']
print('descriptor with material names, targets: ', descriptor)


descriptor.to_csv(path + save_filename, index=False)


print('Done!')





