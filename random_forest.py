import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest.RandomForest import RandomForest
from sklearn.metrics import accuracy_score

data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
data = data[:100]

X = data.drop(columns="OSNR").to_numpy()
y = data["OSNR"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print([ type(x) for x in X_train[0]])
model = RandomForest()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))