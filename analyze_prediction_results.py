print("Hello, world!")

import shutil
import sys
import os
import argparse
import time
import re
from collections import Counter
import itertools

import numpy as np
import pandas as pd
import math
import random
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, ttest_ind
from statsmodels.stats.multicomp import MultiComparison
import scikit_posthocs as sp






parser = argparse.ArgumentParser( description="")


#parser.add_argument("--configuration_type", default="", help="")

#parser.add_argument("--n_value", default="", help="")

parser.add_argument("--file_path", default="", help="")

parser.add_argument("--filename", default="", help="")

parser.add_argument("--save_filename", default="", help="")

#parser.add_argument("--target_prop", default="", help="")

#parser.add_argument("--input_feat", default="", help="")



#change target prop names to not have dates, 'dissertation' in them
def change_target_prop_names(df, target_props_list, old_target_names):

    columns = dict(zip(old_target_names, target_props_list))
    # Rename columns
    df.rename(columns=columns, inplace=True)
    
    return df




def get_after_last(s, char):
    index = s.rfind(char)
    return s[index + len(char):] if index != -1 else ''

def pop_substring(s, substring):
    index = s.find(substring)
    if index == -1:
        return None, s  # substring not found
    new_string = s[:index] + s[index + len(substring):]
    return substring, new_string

def pop_after_second_to_last_integer(s):
    # Find all integers (continuous digit sequences) in the string with their positions
    matches = list(re.finditer(r'\d+', s))
    
    if len(matches) < 2:
        # Not enough integers to pop after the second-to-last one
        return None, s
    
    # Get the second-to-last integer match
    second_last = matches[-2]
    end_index = second_last.end()  # position after the second-to-last integer
    
    # Split the string
    first_part = s[:end_index]
    remainder = s[end_index:]
    return first_part, remainder

def pop_until_integer_1_to_10(s):
    match = re.search(r'(10|[1-9])', s)
    if match:
        end_index = match.end()
        first_part = s[:end_index]
        remainder = s[end_index:]
        return first_part, remainder
    else:
        return None, s  # No integer found between 1 and 10
    
def pop_after_last_char(s, char):
    index = s.rfind(char)
    if index == -1:
        return s, ''  # Character not found: nothing to pop
    return s[:index + 1], s[index + 1:]


def combine_csvs(df1, df2):

    # Read the two CSV files
    #df1 = pd.read_csv(path1)
    #df2 = pd.read_csv(path2)

    # Concatenate the DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Save to a new CSV
    #combined_df.to_csv(save_path, index=False)

    return combined_df

def drop_nums_in_columns(df, columns):
    # Remove digits
    for col in columns:
        df[col] = df[col].str.replace(r'\d+', '', regex=True)

    return df


def get_input_features(MXenes_df, conf_type, n_value, root):

    # get input features for entire predictions list
    # Get input features for all prediction results
    configuration_type_list = []
    n_value_list = []
    M_combination_list = []
    M1_list = []
    M2_list = []
    M3_list = []
    M4_list = []
    X_list = []	
    Tx_list = []
    #fix X issue of Br in X
    for i in range(len(MXenes_df)):
        Material = MXenes_df['Material'].iloc[i]
        Material = Material.replace('.cif', '')    # drop '.cif' from name
        M_combination = get_after_last(root, '/')
        num_hyphens = M_combination.count('-')  # M_combination number of hyphens

        for i in range(4):  # get M1, M2, M3, M4; put N/A if a MXene does not have that many unique M elements
            if i < num_hyphens + 1:
                if i == 0:
                    M1, Material = pop_until_integer_1_to_10(Material)
                elif i == 1:
                    M2, Material = pop_until_integer_1_to_10(Material)
                elif i == 2:
                    M3, Material = pop_until_integer_1_to_10(Material)
                else:  # i == 3
                    M4, Material = pop_until_integer_1_to_10(Material)
            else:
                if i == 0:
                    M1 = 'N/A'
                elif i == 1:
                    M2 = 'N/A'
                elif i == 2:
                    M3 = 'N/A'
                else:  # i == 3
                    M4 = 'N/A'

        #M1, Material = pop_substring(Material, substring)
        #Material, X = pop_after_second_to_last_integer(Material)
        X, Material = pop_until_integer_1_to_10(Material)
        # make sure that this is in X_possibilities_list
        if conf_type == 'ABC' or conf_type == 'ABA':
            Tx = 'none'
        else: 
            #Material, Tx = pop_after_last_char(Material, str(n_value))
            Tx = Material
            # Tx = rest of leftover string
            # make sure that this is in Tx_possibilities_list


        configuration_type_list.append(conf_type)
        n_value_list.append(n_value)
        M_combination_list.append(M_combination)
        M1_list.append(M1)
        M2_list.append(M2)
        M3_list.append(M3)
        M4_list.append(M4)
        X_list.append(X)
        Tx_list.append(Tx)

    MXenes_df.insert(1, 'configuration_type', configuration_type_list)
    MXenes_df.insert(2, 'n_value', n_value_list)
    MXenes_df.insert(3, 'M_combination', M_combination_list)
    MXenes_df.insert(4, 'M1', M1_list)
    MXenes_df.insert(5, 'M2', M2_list)
    MXenes_df.insert(6, 'M3', M3_list)
    MXenes_df.insert(7, 'M4', M4_list)
    MXenes_df.insert(8, 'X', X_list)
    MXenes_df.insert(9, 'Tx', Tx_list)

    return MXenes_df


def get_classification(MXenes_df, new_columns_thresh_dict):
    
    for prop, threshold in new_columns_thresh_dict.items():
        if prop in list(MXenes_df.columns):
            if prop == 'Band_Gap':
                MXenes_df['Band_Gap_thresh'] = MXenes_df[prop].apply(lambda x: 0 if x < threshold else x)
            elif prop == 'Heat_of_Formation':

                choices = ['Highly_stable', 'Stable', 'Metastable', 'Marginally_stable', 'Unstable']
                conditions = [
                    MXenes_df['Heat_of_Formation'] < -1.0,
                    (MXenes_df['Heat_of_Formation'] >= -1.0) & (MXenes_df['Heat_of_Formation'] < -0.5),
                    (MXenes_df['Heat_of_Formation'] >= -0.5) & (MXenes_df['Heat_of_Formation'] < -0.1),
                    (MXenes_df['Heat_of_Formation'] >= -0.1) & (MXenes_df['Heat_of_Formation'] <= 0.0),
                    MXenes_df['Heat_of_Formation'] > 0.0
                ]

                MXenes_df['HoF_stability'] = np.select(conditions, choices)
            else:
                MXenes_df['bool_' + prop] = MXenes_df[prop] < threshold
        else: 
            print('Property ', prop, ' NOT IN DF!!!!')

    return MXenes_df



def get_top_bottom_1000(df, target_props_list):

    selected_indices = set()

    for col in target_props_list:
        top_indices = df.nlargest(1000, col).index
        bottom_indices = df.nsmallest(1000, col).index
        selected_indices.update(top_indices)
        selected_indices.update(bottom_indices)

    # Get all rows with indices in the selected set
    extrema_df = df.loc[list(selected_indices)].copy()

    return extrema_df



def get_target_violin_plots(df, target_prop, save_path, save_filename):

    # make plots containing the training set data distributions for each target property
    # Plot violin plots with overlays

    if target_prop == 'Magnetic.csv' or target_prop == 'Dynamically_stable.csv':
        #df = pd.read_csv(training_dataset_dir + target_prop, header=None)
        sns.countplot(df[target_prop], x=target_prop, order=["True", "False"])
        plt.title('Count of ' + target_prop + ' Classes (T/F)')
        plt.xlabel(target_prop)
        plt.ylabel("Count")
        plt.savefig(save_path + target_prop + '_countplot_' + save_filename, dpi=300, bbox_inches='tight')  # dpi=300 for high resolution
        #plt.show()
    else:
        #df = pd.read_csv(training_dataset_dir + target_prop, header=None)

        plt.figure(figsize=(6, 6))
        sns.violinplot(y=df[target_prop], inner=None, color='lightblue')

        # Statistics
        data = df[target_prop]
        mean = data.mean()
        min_val = data.min()
        max_val = data.max()
        mode = data.mode().iloc[0] if not data.mode().empty else np.nan
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)

        # Plot overlays
        plt.axhline(mean, color='red', linestyle='-', label='Mean')
        plt.axhline(mode, color='purple', linestyle='--', label='Mode')
        plt.axhline(min_val, color='green', linestyle=':', label='Min')
        plt.axhline(max_val, color='green', linestyle=':', label='Max')
        plt.axhline(q25, color='orange', linestyle='-.', label='25th percentile')
        plt.axhline(q75, color='orange', linestyle='-.', label='75th percentile')

        plt.title(f"Violin Plot for {target_prop}")
        plt.ylabel(target_prop)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path + target_prop + '_violin_plot_' + save_filename, dpi=300, bbox_inches='tight')  # dpi=300 for high resolution

        #plt.show()








# create sbatch file to do this with job arrays
    # maybe schedule one job for structure type and n combination
        # 90 jobs





args = parser.parse_args(sys.argv[1:])

#conf_type = args.configuration_type
#n_value = args.n_value
file_path = args.file_path
filename = args.filename
save_filename = args.save_filename
#target_prop = args.target_prop
#input_feat = args.input_feat

#base_dir = file_path + conf_type + '/' + n_value

#filename = "/home/ewvertina/ALIGNNTL/Prediction_Results/additional_predictions.csv"








new_columns_thresh_dict = {'Magnetic':-0.5, 'Band_Gap':0.05, 'Dynamically_stable':-0.5, 'Heat_of_Formation':-1}

input_columns = ['n_value','configuration_type', 'M_combination', 'M1',
       'M2', 'M3', 'M4', 'X', 'Tx']
input_columns_categorical = ['configuration_type', 'M_combination', 'M1',
       'M2', 'M3', 'M4', 'X', 'Tx']

target_props_list = ['Termination_Binding_Energy', 'dBand_Center', 'Density_of_States', 'Heat_of_Formation',
                     'Bulk_Modulus', 'Magnetic', 'Work_Function', 'Band_Gap', 'Dynamically_stable']

old_target_names = ['2025-04-01-Binding_Energy_dissertation', '2025-04-01-dBand_Center_dissertation',
                    '2025-04-01-Density_of_States_dissertation', '2025-04-01-Heat_of_Formation_dissertation',
                    '2025-05-01-Bulk_Modulus_dissertation', '2025-05-01-Magnetic_dissertation',
                    '2025-05-01-Work_Function_dissertation', '2025-05-02-Band_Gap_dissertation',
                    '2025-05-02-Dynamically_stable_dissertation']

#dataset_filename = '/home/ewvertina/ALIGNNTL/Prediction_Results/extrema_results.csv'
#dataset_filename = '/home/ewvertina/ALIGNNTL/Prediction_Results/entire_predictions_set.csv'

columns_to_remove_nums = ['M1', 'M2', 'M3', 'M4', 'X', 'Tx']


structure_types = ['ABC', 'ABC-hM', 'ABC-hX', 'ABC-m', 'ABC-t', 'ABA', 'ABA-h', 'ABA-hX', 'ABA-m', 'ABA-t']




'''
#combine two very large .csv files
path = '/home/ewvertina/ALIGNNTL/Prediction_Results/'
entire_preds_name = 'entire_predictions_set.csv'
additionals_preds_name = 'additional_predictions.csv'
save_name = 'FINAL_PREDS.csv'
df1 = pd.read_csv(path + entire_preds_name, keep_default_na=False)
df2 = pd.read_csv(path + additionals_preds_name, keep_default_na=False)


giant_df = combine_csvs(df1, df2)
giant_df.to_csv(path + save_name, index=False)
'''




'''
# combines all csv files in sub-subfolders
    # e.g. use this to combine all_props and HF_corrected all props
missing_files_num = 0
num_files = 0

# Walk through directory
for root, dirs, files in os.walk(base_dir):
    # Count depth relative to base_dir
    rel_depth = root[len(base_dir):].count(os.sep)
    #print('root: ', root)
    #print('files: ', files)

    if rel_depth == 1:  # sub-subfolder level
        print('root: ', root)
        print('files: ', files)
        #path = root + '/'
        #print('path: ', path)
        if filename not in files:
            print('Ruh-roh, Raggy!')
            print('root: ', root)
            missing_files_num += 1
        else:
            #print('root: ', root)
            #print('files: ', files)
            MXenes_df = pd.read_csv(root + '/' + filename)
            MXenes_df['M2'] = MXenes_df['M2'].astype(str)
            MXenes_df['M3'] = MXenes_df['M3'].astype(str)
            MXenes_df['M4'] = MXenes_df['M4'].astype(str)
            MXenes_df['Tx'] = MXenes_df['Tx'].astype(str)
            #MXenes_df = change_target_prop_names(MXenes_df, target_props_list, old_target_names)         
            #MXenes_df = get_input_features(MXenes_df, conf_type, n_value, root)
            #MXenes_df_no_nums = drop_nums_in_columns(MXenes_df, columns_to_remove_nums)
            #MXenes_df = get_classification(MXenes_df, new_columns_thresh_dict)
            MXenes_df.to_csv(root + '/' + save_filename, index=False)

            #print(MXenes_df)
            #combine_csvs(path + file1, home + save_file, home + save_file)  #use this to combine all_props and HF_corrected all props
            print(root + '/' + save_filename, ' saved!')


            #saves them as specified file

        num_files += 1

print('missing_files_num: ', missing_files_num)
print('num_files combined: ', num_files)
'''




#MXenes_df = change_target_prop_names(filename, target_props_list, old_target_names)


'''
# combine either entire_list .csvs into one or combine additional_preds into one
# will still need to combine these two after this
i = 0
j = 0
num_files = 0
for conf in structure_types:
    base_dir = file_path + conf     
    #print('base_dir: ', base_dir)
    # Walk through directory
    for root, dirs, files in os.walk(base_dir):
        # Count depth relative to base_dir
        rel_depth = root[len(base_dir):].count(os.sep)
        #print('root: ', root)
        #print('files: ', files)

        if rel_depth == 2:  # sub-subfolder level
            #print('root: ', root)
            #print('files: ', files)
            #path = root + '/'
            #print('path: ', path)
            if filename not in files:
                print('Ruh-roh, Raggy!')
                print('root: ', root)
                print('files: ', files)
                i += 1
            elif j == 0:
                entire_MXenes_df = pd.read_csv(root + '/' + filename, keep_default_na=False)
                j += 1
            else:
                #print('root: ', root)
                #print('files: ', files)
                MXenes_df = pd.read_csv(root + '/' + filename, keep_default_na=False)
                #MXenes_df = change_target_prop_names(MXenes_df, target_props_list, old_target_names)
                            
                #MXenes_df = get_input_features(MXenes_df, conf_type, n_value, root)
                
                #MXenes_df_no_nums = drop_nums_in_columns(MXenes_df, columns_to_remove_nums)
                #MXenes_df = get_classification(MXenes_df, new_columns_thresh_dict)
                #the
                #MXenes_df.to_csv(root + '/' + save_filename, index=False)
                #print(MXenes_df)
                entire_MXenes_df = combine_csvs(entire_MXenes_df, MXenes_df)  # use this to combine all_props and HF_corrected all props
                #print('Success!')


                #saves them as specified file

            num_files += 1


print('i: ', i)
print('num_files combined: ', num_files)

entire_MXenes_df.to_csv(file_path + save_filename, index=False)

'''


#MXenes_df_no_nums = drop_nums_in_columns(MXenes_df, columns_to_remove_nums)



# append additional columns to df to have T/F for Magnetic, Dynamic Stability, and Heat of Formation 
    # (and maybe for band gap to assign many of the predicitons to 0 if they are close to 0 and leave as is otherwise?)
#if 'Magnetic' in list(df.columns): #if predicting whether or not a material is magnetic
#    df['Magnetic'] = df['Magnetic'] < -0.5
#
#if 'Dynamically_stable' in list(df.columns): #if predicting whether or not a material is dynamically stable
#    df['Dynamically_stable'] = df['Dynamically_stable'] < -0.5


'''
# drop .csv from 'Material' column in df
MXenes_df = pd.read_csv(file_path + filename, keep_default_na=False)
MXenes_df['Material'] = MXenes_df['Material'].str.replace('.cif', '', regex=False)
MXenes_df.to_csv(file_path + save_filename, index=False)
'''


#path1 = "/home/ewvertina/ALIGNNTL/Prediction_Results/entire_predictions_set.csv"
#path2 = "/home/ewvertina/ALIGNNTL/Prediction_Results/additional_predictions.csv"
#save_path = "/home/ewvertina/ALIGNNTL/Prediction_Results/FINAL_PREDICTIONS.csv"

#combine_csvs(path1, path2, save_path)



'''
# get extrema

MXenes_df = pd.read_csv(file_path + filename, keep_default_na=False)
extrema_df = get_top_bottom_1000(MXenes_df, target_props_list)
extrema_df.to_csv(file_path + save_filename, index=False)
'''



# get only single M MXenes, save it
'''
target_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
MXenes_df = pd.read_csv(file_path + filename, keep_default_na=False)
MXenes_df = MXenes_df[MXenes_df['M_combination'].isin(target_names)]
MXenes_df.to_csv(file_path + save_filename, index=False)
'''

# run statistical tests, but maybe in a different .py file and .sh file
'''
MXenes_df = pd.read_csv(file_path + filename, keep_default_na=False, low_memory=False)
get_target_violin_plots(MXenes_df, target_prop, file_path, save_filename)
'''


'''
#get violin plot matrices by n for each target property
def get_violin_by_n(MXenes_df, prop, n_values_list, save_path):

    # make a plot matrix containing the training set data distributions for each n for a specified property
    n_rows = 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    # Plot violin plots with overlays
    i = 0
    j = 1

    for n_value in n_values_list:
        df = MXenes_df[MXenes_df['n_value'] == n_value]
        plt.figure(figsize=(6, 6))
        sns.violinplot(y=df[prop], inner=None, color='lightblue', ax=axes[math.floor(i/3), i%3])
        # Statistics
        data = df[prop]
        mean = data.mean()
        min_val = data.min()
        max_val = data.max()
        mode = data.mode().iloc[0] if not data.mode().empty else np.nan
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)

        # Plot overlays
        axes[math.floor(i/3), i%3].axhline(mean, color='red', linestyle='-', label='Mean')
        axes[math.floor(i/3), i%3].axhline(mode, color='purple', linestyle='--', label='Mode')
        axes[math.floor(i/3), i%3].axhline(min_val, color='green', linestyle=':', label='Min')
        axes[math.floor(i/3), i%3].axhline(max_val, color='green', linestyle=':', label='Max')
        axes[math.floor(i/3), i%3].axhline(q25, color='orange', linestyle='-.', label='25th percentile')
        axes[math.floor(i/3), i%3].axhline(q75, color='orange', linestyle='-.', label='75th percentile')

        title = 'n = ' + n_value[1]
        subtitle = f'{len(df):,}' + ' MXenes'
        axes[math.floor(i/3), i%3].set_title(f"{title}\n{subtitle}")
        axes[math.floor(i/3), i%3].set_xlabel('')
        axes[math.floor(i/3), i%3].set_ylabel(prop.replace("_", " "))

        if i == 0:
            handles, labels = axes[0, 0].get_legend_handles_labels()
        i += 1

    fig.suptitle(prop.replace("_", " ") + " Distribution by n", y = 0.95, verticalalignment = 'center', fontsize=28)
    fig.legend(handles=handles, labels=labels, loc='upper right', fontsize=12)

    
    fig.savefig(save_path + prop + '_violin_plots_by_n.png', dpi=300)


print('target_prop: ', target_prop)
n_values_list = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']
#MXenes_df = pd.read_csv(file_path + filename, keep_default_na=False)
#print('MXenes_df.columns: ', MXenes_df.columns)
MXenes_df = pd.read_csv(file_path + filename, usecols=['n_value', target_prop], keep_default_na=False)
df = MXenes_df[['n_value', target_prop]]
get_violin_by_n(MXenes_df, target_prop, n_values_list, file_path + save_filename)
'''






'''
# make plots by input feature, target property


def get_input_plots(MXenes_df, input_feat, target_prop, save_path): #  Generate plots of average values of each target property grouped by each categorical column, including std dev as error bars

    sns.set_theme(style="whitegrid")
    # Group and compute mean and std
    grouped = MXenes_df.groupby(input_feat)[target_prop].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values(by='mean', ascending=True)  # sort the df

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=grouped,
        x=input_feat,
        y='mean',
        yerr=grouped['std'],
        capsize=0.2,
        color='skyblue'
    )

    # Customize plot
    #ax.set_title(f'Mean and Std Dev of {prop} by {cat_col}')
    #ax.set_ylabel(f'Mean {prop}')
    ax.set_title(f'Mean and Std Dev of {target_prop.replace("_", " ")} by {input_feat.replace("_", " ")}')
    ax.set_ylabel(f'Mean {target_prop.replace("_", " ")}')
    #ax.set_xlabel(cat_col)
    ax.set_xlabel(input_feat.replace("_", " "))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    filename = save_path + 'average_' + target_prop + '_by_' + input_feat + '.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f" Saved: {filename}")

    return 'Done!'






MXenes_df = pd.read_csv(file_path + filename, usecols=[input_feat, target_prop], keep_default_na=False)
get_input_plots(MXenes_df, input_feat, target_prop, file_path + save_filename)
'''


'''
# Dunn's test, make plots -> NOOOOOOOOOOOO; need Duncan's test

def Dunns_test(MXenes_df,  input_feat, target_prop, save_path):
    # Use DMRT approximation (Dunn test with Bonferroni correction is more conservative, but similar style)
    result = sp.posthoc_dunn(MXenes_df, val_col=target_prop, group_col=input_feat, p_adjust='bonferroni')
    print(result)

    plt.figure(figsize=(8, 6))
    sns.heatmap(result, annot=True, cmap='coolwarm', fmt=".1f", cbar_kws={'label': 'p-value'})
    plt.title(target_prop + " " + input_feat + " Dunn's Test (Bonferroni-adjusted p-values)")
    plt.show()
    plt.savefig(save_path + target_prop + '_' + input_feat + '.png', dpi=300)
    plt.close()
    print(f" Saved: {filename}")



MXenes_df = pd.read_csv(file_path + filename, usecols=[input_feat, target_prop], keep_default_na=False)
Dunns_test(MXenes_df,  input_feat, target_prop, file_path + save_filename)
'''




'''
# count num of MXenes with at least one stability measure satisfied
file = '/home/ewvertina/ALIGNNTL/Prediction_Results/FINAL_PREDS.csv'
df = pd.read_csv(file, keep_default_na=False)

# Define your condition — for example, keep rows where A, B, or C > 5
filtered_df = df[(df['bool_Dynamically_stable'] == True) | (df['HoF_stability'] == 'Metastable') | (df['HoF_stability'] == 'Marginally_stable') | (df['HoF_stability'] == 'Stable') | (df['HoF_stability'] == 'Highly_stable')]

# Number of rows after filtering
num_rows = len(filtered_df)

print(filtered_df)
print("Number of rows after filtering:", num_rows)
'''





# Run Duncan's test
'''
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns

# Example dataset
MXenes_df = 

# Perform Duncan's test: e.g., total_bill by day
result = sp.posthoc_duncan(MXenes_df, val_col='Work_Function', group_col='X')

print(result)




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from statsmodels.stats.multicomp import MultiComparison
import numpy as np

# Load example data
df = sns.load_dataset("tips")

# Run Duncan's test
duncan_res = sp.posthoc_duncan(df, val_col='total_bill', group_col='day')

# Calculate group means for plotting
group_means = df.groupby('day')['total_bill'].mean().sort_values()

# Step 1: Generate compact letter display (manual approach)
# This function determines letter groupings from a p-value matrix
def get_cld(pval_df, alpha=0.05):
    from collections import defaultdict
    import itertools

    groups = list(pval_df.columns)
    letters = defaultdict(str)
    assigned = []

    for i, g in enumerate(groups):
        # Start a new group
        current_letter = chr(97 + i)  # a, b, c, ...
        compatible = [g]
        for h in groups:
            if g == h:
                continue
            if pval_df.loc[g, h] > alpha:
                compatible.append(h)
        # Assign letter to all compatible groups not already assigned
        for c in compatible:
            if current_letter not in letters[c]:
                letters[c] += current_letter
        assigned.append(compatible)
    return letters

# Apply CLD grouping
cld = get_cld(duncan_res)

# Prepare data for plotting
means_df = group_means.reset_index()
means_df['letters'] = means_df['day'].map(cld)

# Step 2: Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=means_df, x='day', y='total_bill', palette='pastel')

# Add significance letters above bars
for i, row in means_df.iterrows():
    plt.text(i, row['total_bill'] + 1, row['letters'], ha='center', va='bottom', fontsize=12)

plt.title("Mean Total Bill by Day with Duncan Grouping")
plt.ylabel("Mean Total Bill")
plt.xlabel("Day")
plt.tight_layout()
plt.show()



'''





'''
# get bar chart plot matrixes by target property


def get_input_plots(df, input_columns, target_prop, save_path): #  Generate plots of average values of each target property grouped by each categorical column, including std dev as error bars
    
    # make a plot matrix containing the training set data distributions for each n for a specified property
    n_rows = 2
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))

    i = 0
    for input_feat in input_columns:
        if input_feat == 'M2' or input_feat == 'M3' or input_feat == 'M4':
            pass
        else:
            MXenes_df = df[[input_feat, target_prop]]
            #if input_feat == 'M_combination':
                #MXenes_df['M_combination'] = MXenes_df['M_combination'].astype(str)
            plt.figure(figsize=(7, 8))
            
            sns.set_theme(style="whitegrid")
            # Group and compute mean and std
            grouped = MXenes_df.groupby(input_feat)[target_prop].agg(['mean', 'std']).reset_index()
            #print('grouped after .groupby: ', grouped)
            grouped = grouped.sort_values(by='mean', ascending=True)  # sort the df
            #print('grouped after sorting: ', grouped)
            grouped = grouped[grouped[input_feat] != 'nan']
            #print('grouped after dropping nan: ', grouped)
            sns.barplot(
                data=grouped,
                x=input_feat,
                y='mean',
                yerr=grouped['std'],
                capsize=0.2,
                color='skyblue',
                ax=axes[math.floor(i/3), i%3]
            )

            title = input_feat.replace("_", " ")
            subtitle = f'{len(df):,}' + ' MXenes'
            axes[math.floor(i/3), i%3].set_title(f"{title}\n{subtitle}")
            #axes[math.floor(i/3), i%3].set_title(f'{input_feat.replace("_", " ")}')
            axes[math.floor(i/3), i%3].set_xlabel('')
            if input_feat == 'M1' or input_feat == 'M2' or input_feat == 'M3' or input_feat == 'M4':
                axes[math.floor(i/3), i%3].tick_params(axis='x', labelsize=5)
            elif input_feat == 'M_combination':
                axes[math.floor(i/3), i%3].tick_params(axis='x', labelsize=4)
            elif input_feat == 'n_value' or input_feat == 'X' :
                axes[math.floor(i/3), i%3].tick_params(axis='x', labelsize=12)
            elif input_feat == 'configuration_type':
                axes[math.floor(i/3), i%3].tick_params(axis='x', labelsize=10)
            else:
                axes[math.floor(i/3), i%3].tick_params(axis='x', labelsize=8)
            axes[math.floor(i/3), i%3].set_xticklabels(axes[math.floor(i/3), i%3].get_xticklabels(), rotation=90)
            axes[math.floor(i/3), i%3].set_ylabel(f'Mean {target_prop.replace("_", " ")}')
            
            if i == 0:
                handles, labels = axes[0, 0].get_legend_handles_labels()
            i += 1


    fig.suptitle("Mean " + target_prop.replace("_", " ") + " Predictions by input feature", y = 0.95, verticalalignment = 'center', fontsize=28)
    #fig.legend(handles=handles, labels=labels, loc='upper right', fontsize=12)
    plt.show()
    
    fig.savefig(save_path + target_prop + '_plots_by_input_feature.png', dpi=300)

    print(f" Saved: {filename}")

    return 'Done!'


MXenes_df = pd.read_csv(file_path + filename, usecols=input_columns + [target_prop], keep_default_na=False, dtype={'M_combination':str})
#MXenes_df['M_combination'] = MXenes_df['M_combination'].astype(str)
get_input_plots(MXenes_df, input_columns, target_prop, file_path + save_filename)

'''






selected_cols = ['bool_Dynamically_stable', 'HoF_stability']
MXenes_df = pd.read_csv(file_path + filename, usecols=['Band_Gap_thresh', 'bool_Dynamically_stable', 'HoF_stability'], keep_default_na=False, dtype={'bool_Dynamically_stable':str})

category_counts = MXenes_df[selected_cols].apply(pd.value_counts).fillna(0).astype(int)

print('category counts: ', category_counts)

num_zero_rows = MXenes_df[MXenes_df['Band_Gap_thresh'] == 0].shape[0]

print('num band gap = 0: ', num_zero_rows)

print('num non-zero band gap: ', len(MXenes_df) - num_zero_rows)










