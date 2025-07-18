print("Hello, world!")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import argparse
import sys
import json
import joblib






parser = argparse.ArgumentParser(description="")



parser.add_argument("--path", default="", help="")

parser.add_argument("--target_prop", default="", help="")

parser.add_argument("--best_model", default="", help="")

parser.add_argument("--seed_num", default="", help="")

parser.add_argument("--baseline_model", default="", help="")

parser.add_argument("--data_set_filename", default="", help="")

parser.add_argument("--saved_model_filename", default="", help="")







class FeedForwardNN(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)



def pred_neural_network(X, y, saved_model_path, save_path):


    # Convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    X_test  = torch.tensor(X.values, dtype=torch.float32).to(device)
    #print(X_test)
    #nn_model = FeedForwardNN(X_test.shape[1]).to(device)
    nn_model = FeedForwardNN(290).to(device)



    # Load the saved weights
    nn_model.load_state_dict(torch.load(saved_model_path))


    for param in nn_model.parameters():
        print(param.mean(), param.std())


    # Set to eval mode (important!)
    nn_model.eval()

    with torch.no_grad():
        nn_preds = nn_model(X_test).numpy().flatten().tolist()

    print('predictions: ', nn_preds)

    return nn_preds


def pred_linear_regression(X, y, saved_model_path, save_path):

    loaded_model = joblib.load(saved_model_path)
    predictions = loaded_model.predict(X)
    predictions = [pred[0] for pred in predictions]
    print('predictions: ', predictions)

    return predictions


def pred_random_forest(X, y, saved_model_path, save_path):

    loaded_model = joblib.load(saved_model_path)
    predictions = loaded_model.predict(X)
    predictions = predictions.tolist()
    print('predictions: ', predictions)

    return predictions


def pred_xgboost(X, y, saved_model_path, save_path):

    model = joblib.load(saved_model_path)
    # Predict
    predictions = model.predict(X)
    predictions = predictions.tolist()
    print('predictions: ', predictions)

    return predictions





'''
path = file_path

target_prop = 'Binding_Energy'
best_model = best_model_name
seed_num = '3'
baseline_model = "NeuralNetwork"   # ["LinearRegression", "RandomForest", "XGBoost", "NeuralNetwork"]
data_set_filename = "xenonpy_Binding_Energy_final_holdout_set.csv"

'''

args = parser.parse_args(sys.argv[1:])

path = args.path

target_prop = args.target_prop
best_model = args.best_model
seed_num = args.seed_num
baseline_model = args.baseline_model
data_set_filename = args.data_set_filename
saved_model_filename = args.saved_model_filename
save_path = save_path
saved_model_path = saved_model_path

material_filename = 'MXene_filename'
target_prop_column = 'Target'

final_holdout_set = pd.read_csv(save_path + data_set_filename)

X = final_holdout_set.loc[:, ~final_holdout_set.columns.isin([material_filename, target_prop_column])]
y = final_holdout_set[target_prop_column].to_list()

names = final_holdout_set[material_filename].to_list()





if baseline_model == "XGBoost":
    predictions = pred_xgboost(X, y, saved_model_path, save_path)
elif baseline_model == "RandomForest":
    predictions = pred_random_forest(X, y, saved_model_path, save_path)
elif baseline_model == "LinearRegression":
    predictions = pred_linear_regression(X, y, saved_model_path, save_path)
elif baseline_model == "NeuralNetwork":
    predictions = pred_neural_network(X, y, saved_model_path, save_path)
else:
    print('Check model type passed!')





preds_data = {material_filename:names, 'y_true':y, 'y_pred':predictions}
preds_data_df = pd.DataFrame(preds_data)
if target_prop in ['Magnetic', 'Dynamically_stable']:
    preds_data_df['y_pred'] = (preds_data_df['y_pred'] >= 0.5).astype(int)
print('preds_data_df: ', preds_data_df)


preds_data_df.to_csv(save_path, index=False)








