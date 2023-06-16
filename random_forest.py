import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest.RandomForest import RandomForest
from sklearn.metrics import mean_squared_error

unbound_columns = ["transponder_bitrate", "transponder_modulation", "Unnamed: 0"]

data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
for u in unbound_columns:
    if u in data.columns:
        data = data.drop(columns=u)
data = data[:100]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print([ type(x) for x in X_train[0]])
model = RandomForest(num_trees=5, min_samples_split=2, max_depth=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(mean_squared_error(y_test, preds))