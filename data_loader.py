"""
data_loader.py
--------------
Build data pipeline
Load CSV and Labels
Tag each row 0/1
"""

import json
import numpy as np
import pandas as pd

# opens combined_windows file, loads it as Python dictionary. Each key is a filename and each value is a list (start,end) marking the anomaly windows
def load_labels(labels_path="data/labels/combined_windows.json"):
    with open(labels_path, "r") as f:
        return json.load(f)

#Reads CSV into table with timestamp/value columns 
#new label column is added, default is 0
#For each anomaly, it finds all rows whose timestamp is in between windows start and end and replaces its label to 1
def load_series(csv_path, file_key, labels_dict):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["label"] = 0
    windows = labels_dict.get(file_key, [])
    for start_str, end_str in windows:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
        df.loc[mask, "label"] = 1

    return df

#z-score normalizes values - subtract mean, divide by standard deviation
def normalize(values):
    mean = values.mean()
    std = values.std()
    if std == 0:
        std = 1e-8 #prevents division by zero error
    return (values - mean) / std, mean, std

#turns 1D array of values into input/target sequences for next step forecasting
def make_sequences(values, window_size):
    X, y, idx = [], [], [] #X:features(inputs for the model), y:target label(value model should predict), idx:lists of predicted values (tracking)

    for i in range(len(values) - window_size):
        X.append(values[i:i + window_size]) #takes portion of length and saves it to X
        y.append(values[i + window_size])#takes single value after that portion to be label y
        idx.append(i + window_size)#stores index position of y

    X = np.array(X).reshape(-1, window_size, 1)#reshapes X into 3D shape
    y = np.array(y).reshape(-1, 1) #reshapes y into 2d column 
    return X, y, np.array(idx)



