import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest.RandomForest import RandomForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.cross_validation import cross_validation
import time

unbound_columns = ["transponder_bitrate", "transponder_modulation", "Unnamed: 0"]

data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
for u in unbound_columns:
    if u in data.columns:
        data = data.drop(columns=u)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_forest = RandomForest(num_trees=5, min_samples_split=2, max_depth=3, verbose=True)

st = time.time()
mean, std = cross_validation(model_forest,
                       data.drop(columns="OSNR"),
                       data["OSNR"],
                       3,
                       mean_absolute_error,
                       [np.mean, np.std],
                       verbose=True)
end = time.time()
print(f"On cross validation, the algorithm scores {mean:.3f} ± {std:.3f} mean absolute error")
print(end-st)