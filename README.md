# Read Me


# Desired MXene target properties

|Property Name|Property Name (continued)|
|----------|----------|
|Band Gap|Magnetic (T/F)|
|Carrier Density|Metallic (T/F)|
|Carrier Effective Mass|Optical Absorption Coefficient|
|Carrier Lifetime|Permeability|
|Carrier Mobility|Permittivity|
|Charge Recombination|Poisson’s Ratio|
|d-Band Center|Reflectivity|
|Density of States at Fermi Level|Resistivity|
|Dynamically Stable|Seebeck Coefficient |
|Elastic Moduli (Bulk, Shear, Young’s)|Sensitivity to Disorder|
|Elastic Coefficients, c11 and c12|(Termination) Binding Energy|
|Energy above Convex Hull|Thermal Conductivity|
|Electrical Conductivity|THz Range (Transmission)|
|Heat of Formation|Transmittance|
|Hole Effective Mass|Work Function|
|Lattice Constant||



# MXene elements, structure types considered
MXene M elements we consider
![Alt text]( Figures/periodic_table_M.png)

MXene X elements we consider
![Alt text]( Figures/periodic_table_X.png)

MXene Tx (surface terminations) we consider
![Alt text]( Figures/periodic_table_Tx.png)


MXene structure types we consider:
![Alt text]( Figures/MXene_structure_types.png)

# Our workflow
![Alt text]( Figures/flowchart_smaller.png)

# Prediction results for 13 million MXenes
The following are violin plots for each of the 9 target properties that we predicted across all 13 million+ unique MXenes that we generated.
![Alt text]( Figures/violin_all_props_all_n_smaller.png)





# Using this code



Create conda environment to run this code:
conda create --name env_name python=3.8
source activate env_name

Might need to install (the appropriate versions to get matching CUDA versions):
conda install dgl=2.1.0 pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia

conda install dgl=0.6.0 pytorch=1.7.1 torchvision=0.10.1 torchaudio=0.10.0 pytorch-cuda=10.1 -c pytorch -c nvidia
	(Check which versions to install and put them after each = sign, respectively)

Packages installed:
pip install numpy==1.19.5

pip install torch==1.8.0
	num-py
pip install scipy==1.6.1

pip install dgl-cu101==0.6.1
	sci-py
pip install dgl==0.6.1

pip install matplotlib=3.4.2

pip install tqdm==4.60.0

pip install pyparsing==2.2.1

pip install jarvis-tools==2021.07.19
	matplotlib
	tqdm
	pyparsing
pip install pytorch-ignite==0.4.7

pip install pydantic==1.8.1

pip install pycodestyle==2.7.0

pip install flake8==3.9.1
	pycodestyle
pip install pydocstyle==6.0.0


conda install pandas=1.2.3
	num-py

conda install matplotlib=3.4.2

conda install scikit-learn=0.23




conda install pandas=1.2.3

conda install matplotlib=3.4.2

conda install scikit-learn=0.23

pip install --force-reinstall charset-normalizer==2.0.4



Ran this in ALIGNNTL:
python -m pip install -e .
	might not have worked

Ran this in ALIGNNTL/FineTuning:
python -m pip install -e .
	Note: I copied the original setup.py file from .. into FineTuning











Steps for using ALIGNNTL such that I specify a source dataset that is similar to (probably has to have same target property) as a specified MXene property I want to predict:
	
 1. If not super large dataset, copy POSCAR/CIF/etc. files into a directory for the source dataset here:
		/path/ALIGNNTL/SOURCE_DIRECTORY
	
 2. Copy the .csv target properties for the dataset into the source dataset directory such that there are two columns, the first one being the name of the POSCAR/CIF/etc. file for the corresponding record, and the second being the value of the target property
	
 3. Copy the default config file into this source dataset directory
	
 4. Modify the config file to make sure that train n, val n, and test n are correct. 
	
 5. Change source_model to be the pretrained model for the specific property. If there isn't one, change this to "null" (if it isn't already null)
		a. Can even change "target"? If you know it??? Not sure about this, though
	
 6. Make sure that the batch size is a multiple of the training, test, val sets (or, at the very least, that it is smaller than these sets)
	
 7. Make sure that the number of epochs is small enough to do a quick run to see if it works (e.g. 3)
	
 8. Run this code, making sure that the correct root_dir, config file, id_prop_file, and output directory are correctly specified
	
 9. In VS code, open the ALIGNNTL directory in the explorer taskbar to the left
	10. Download the output directory from this source training task onto my laptop
		a. Perhaps then delete .pt files until there is only one .pt file remaining?
	
 11. Open the file directory on my laptop, go to this downloaded directory, and zip this folder
	
 12. Move this .zip folder onto Turing by dragging its icon from the file directory onto VS code and place it into this directory:
		/path/ALIGNNTL/FineTuning/alignn/DIRECTORY_NAME.zip
	
 13. For my target MXenes dataset, change the config file's source model to have "DIRECTORY_NAME" as "source_model"
	
 14. Make sure that n_val, n_test, and n_train are correctly set
	
 15. Make sure that the batch size is smaller than the training, test, and val sets
	
 16. Make the number of epochs small just to make sure that this runs properly
	
 17. Edit the train.py file here to add the pretrained model that we just trained to the list of pretrained models. This file can be found here: /path/ALIGNNTL/FineTuning/alignn/train.py   . Code look something like this (if DIRECTORY_NAME is "2024-07-20"):
	
 BEFORE:
	all_models = {
	    "mp_e_form_alignnn": [
	        "https://figshare.com/ndownloader/files/31458811",
	        1,
	    ],
	    "mp_gappbe_alignnn": [
	        "https://figshare.com/ndownloader/files/31458814",
	        1,
	    ],
	    "jv_bulk_modulus_kv_alignn": [
	        "https://figshare.com/ndownloader/files/31458649",
	        1,
	    ],
	    }

AFTER:
all_models = {
    "mp_e_form_alignnn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignnn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "2024-07-20": [
        "",
        1,
    ],
}
	


18. Run this code, making sure that the correct root_dir, config file, id_prop_file, and output directory are correctly specified. Predictions done!!








# My Works

PhD Dissertation: (link to dissertation coming soon!)

Papers:

1. Eric Warnemunde Vertina, Emily Sutherland, Sajal Chakroborty, N. Aaron Deskins, and Oren Mangoubi. Machine-Learning-Enabled High-Throughput MXene Property Prediction. IEEE NAP 2025. (Accepted for publication).

2. Eric Warnemunde Vertina, Sajal Chakroborty, Emily Sutherland, N. Aaron Deskins, and Oren Mangoubi. MXene Material Property Prediction via Transfer Learning
with Graph Neural Networks. IEEE ICAD 2025. https://ieeexplore.ieee.org/document/11114063

3. Eric Warnemunde Vertina, Emily Sutherland, N. Aaron Deskins, and Oren Mangoubi. “MXene Property Prediction via Graph Contrastive Learning.” IEEE NAP 2024. https://ieeexplore.ieee.org/abstract/document/10739742

4. Eric Warnemunde Vertina, Emily Sutherland, N. Aaron Deskins, and Oren Mangoubi. “Predicting MXene Properties via Machine Learning.” In Proceedings of the 2022 International Conference on Machine Learning and Applications (ICMLA). https://ieeexplore.ieee.org/abstract/document/10069589

