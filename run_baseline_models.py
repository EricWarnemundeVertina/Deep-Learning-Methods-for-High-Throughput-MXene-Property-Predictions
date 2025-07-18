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

parser.add_argument("--ids_filename", default="", help="")

parser.add_argument("--target_prop", default="", help="")

parser.add_argument("--best_model", default="", help="")

parser.add_argument("--seed_num", default="", help="")

parser.add_argument("--baseline_model", default="", help="")

parser.add_argument("--data_set_filename", default="", help="")

parser.add_argument("--convert_training_set_mats_filename", default="", help="")






# --- Evaluation ---
def evaluate(y_true, y_pred, model_name):
    print('y_true: ', y_true)
    print('y_pred: ', y_pred)
    evaluation_results = []
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100  # avoid div/0
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name}:")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    print("-" * 30)

    evaluation_results.append({
        "Model": model_name,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R2": r2
    })

    return evaluation_results


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
    


    
# --- PyTorch: Feed-forward Neural Network ---
def run_neural_network(X_train, y_train, X_test, y_test, save_model, save_path):

    #loss_history = []


    # Convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    #X_val_t   = torch.tensor(X_val, dtype=torch.float32)
    #y_val_t   = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_t  = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

    nn_model = FeedForwardNN(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        nn_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = nn_model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if epoch > 900:
                #loss_history.append(loss.item())  # ← store the loss
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    #plt.figure(figsize=(8, 4))
    #plt.plot(loss_history, label='Training Loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.title('Neural Network Training Loss')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    # Predict with NN
    nn_model.eval()
    with torch.no_grad():
        #nn_preds = nn_model(X_test_t.to(device)).numpy().flatten()

        #nn_preds = nn_model(X_test_t.to(device)).cpu().numpy().flatten()
        nn_preds = nn_model(X_test_t).numpy().flatten().tolist()

        #outputs = nn_model(X_test_t.to(device))
        #nn_preds = outputs.cpu().numpy().flatten()

    #print('nn_preds: ', nn_preds.tolist())
    evaluation_results = evaluate(y_test.values.ravel(), nn_preds, "NeuralNetwork")
    if save_model:
        print('About to save model here: ', save_path)
        torch.save(nn_model.state_dict(), save_path)
        print('(Hopefully) the model is saved!')

    return evaluation_results, y_test, nn_preds



# --- Scikit-learn: Linear Regression ---
def run_linear_regression(X_train, y_train, X_test, y_test, save_model, save_path):

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    evaluation_results = evaluate(y_test, lr_preds, "LinearRegression")
    if save_model:
        print('About to save model here: ', save_path)
        joblib.dump(lr_model, save_path)
        print('(Hopefully) the model is saved!')
    lr_preds = [pred[0] for pred in lr_preds]
    return evaluation_results, y_test, lr_preds

# --- Scikit-learn: Random Forest ---
def run_random_forest(X_train, y_train, X_test, y_test, save_model, save_path):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    evaluation_results = evaluate(y_test.values.ravel(), rf_preds.ravel(), "RandomForest")
    if save_model:
        print('About to save model here: ', save_path)
        joblib.dump(rf_model, save_path)
        print('(Hopefully) the model is saved!')
    rf_preds = rf_preds.tolist()
    return evaluation_results, y_test, rf_preds

# --- XGBoost ---
def run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, target_prop, save_model, save_path):
    if target_prop in ['Dynamically_stable', 'Magnetic']:
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, early_stopping_rounds=10)
    else:
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, early_stopping_rounds=10)

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_preds = xgb_model.predict(X_test)
    evaluation_results = evaluate(y_test.values.ravel(), xgb_preds.ravel(), "XGBoost")
    if save_model:
        print('About to save model here: ', save_path)
        joblib.dump(xgb_model, save_path)
        print('(Hopefully) the model is saved!')
    xgb_preds = xgb_preds.tolist()
    return evaluation_results, y_test, xgb_preds


def save_results_to_csv(filename, evaluation_results):
    df = pd.DataFrame(evaluation_results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def get_ids(path_plus_file, baseline_model):
    
    with open(path_plus_file, 'r') as f:
        data = json.load(f)
    train_ids = data['id_train']
    test_ids = data['id_test']
    val_ids = data['id_val']
    if not baseline_model == 'XGBoost':
        train_ids = train_ids + val_ids
    return train_ids, val_ids, test_ids


def change_mat_names(old_df, mapping_df):
    
    # Create a dictionary mapping from mapping_df
    mapping_dict = dict(zip(mapping_df['old_filename'], mapping_df['MXene_Formula']))

    # Update old_df using the mapping
    old_df['MXene_filename'] = old_df['MXene_filename'].map(mapping_dict).combine_first(old_df['MXene_filename'])

    return old_df



'''
path = '/home/ewvertina/ALIGNNTL/Experiment_Results/'
ids_filename = 'ids_train_val_test.json'
target_prop = 'Binding_Energy'
best_model = '2025-04-01'
seed_num = '3'
baseline_model = "NeuralNetwork"   # ["LinearRegression", "RandomForest", "XGBoost", "NeuralNetwork"]
data_set_filename = "xenonpy_Binding_Energy_training_set.csv"
convert_training_set_mats_filename = 'convert_training_set_mat_names2.csv'
'''

args = parser.parse_args(sys.argv[1:])



path = args.path
ids_filename = args.ids_filename
target_prop = args.target_prop
best_model = args.best_model
seed_num = args.seed_num
baseline_model = args.baseline_model
data_set_filename = args.data_set_filename
convert_training_set_mats_filename = args.convert_training_set_mats_filename


target_prop_column = 'Target'
props_new_names_list = ["Band_Gap", "Binding_Energy", "Bulk_Modulus", "dBand_Center", "Density_of_States", 
                        "Dynamically_stable", "Magnetic", "Work_Function"]








convert_training_set_mats_df = pd.read_csv(convert_training_set_mats_filename)
dataset_df = pd.read_csv(path + best_model + '/' + 'C2DB_' + target_prop + '/' + data_set_filename)

if target_prop in props_new_names_list:
    dataset_df = change_mat_names(dataset_df, convert_training_set_mats_df)


print('target_property: ', target_prop)
train_ids, val_ids, test_ids = get_ids(path + best_model + '/' + 'C2DB_' + target_prop + '/' + seed_num + '/' + ids_filename, baseline_model)


train_df = dataset_df[dataset_df['MXene_filename'].isin(train_ids)]
val_df = dataset_df[dataset_df['MXene_filename'].isin(val_ids)]
test_df = dataset_df[dataset_df['MXene_filename'].isin(test_ids)]

# get train, test, val sets and split into inputs X and target y for each
X_train = train_df.loc[:, train_df.columns != target_prop_column]
y_train = train_df[['MXene_filename', target_prop_column]]

X_train = X_train.drop('MXene_filename', axis=1)
y_train = y_train.drop('MXene_filename', axis=1)

if baseline_model == 'XGBoost': # other models don't use val sets, just train and test sets
    X_val = val_df.loc[:, val_df.columns != target_prop_column]
    y_val= val_df[['MXene_filename', target_prop_column]]

    X_val = X_val.drop('MXene_filename', axis=1)
    y_val = y_val.drop('MXene_filename', axis=1)

X_test = test_df.loc[:, test_df.columns != target_prop_column]
y_test= test_df[['MXene_filename', target_prop_column]]

test_names = y_test['MXene_filename'].to_list()
X_test = X_test.drop('MXene_filename', axis=1)
y_test = y_test.drop('MXene_filename', axis=1)






save_model = False
save_path = ''
if seed_num == '6' or (seed_num == '5' and target_prop == 'Band_Gap' and best_model == '2025-05-02' and baseline_model == 'LinearRegression'):
    save_model = True
    save_path = path + best_model + '/' + 'C2DB_' + target_prop + '/' + seed_num + '/' + baseline_model


if baseline_model == "XGBoost":
    evaluation_results, y_true, y_pred = run_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, target_prop, save_model, save_path + '_model.pkl')
elif baseline_model == "RandomForest":
    evaluation_results, y_true, y_pred = run_random_forest(X_train, y_train, X_test, y_test, save_model, save_path + '_model.pkl')
elif baseline_model == "LinearRegression":
    evaluation_results, y_true, y_pred = run_linear_regression(X_train, y_train, X_test, y_test, save_model, save_path + '_model.pkl')
elif baseline_model == "NeuralNetwork":
    evaluation_results, y_true, y_pred = run_neural_network(X_train, y_train, X_test, y_test, save_model, save_path + '_model_weights.pth')
else:
    print('Check model type passed!')


# save predicted values
print('len(test_names): ', len(test_names))
print('len(y_true): ', len(y_true))
print('len(y_pred): ', len(y_pred))

print('type(test_names): ', type(test_names))
print('type(y_true): ', type(y_true))
print('type(y_pred): ', type(y_pred))
preds_dict = {'MXene_filename':test_names, 'y_true':y_test['Target'].to_list(), 'y_pred':y_pred }
preds_df = pd.DataFrame(preds_dict)
if target_prop in ['Magnetic', 'Dynamically_stable']:
    preds_df['y_pred'] = (preds_df['y_pred'] >= 0.5).astype(int)
preds_df.to_csv(path + best_model + '/' + 'C2DB_' + target_prop + '/' + seed_num + '/' + baseline_model + '_preds.csv', index=False)

save_results_to_csv(path + best_model + '/' + 'C2DB_' + target_prop + '/' + seed_num + '/' +  baseline_model + '_results.csv', evaluation_results)

print('Evaluation Results: ', evaluation_results)


















