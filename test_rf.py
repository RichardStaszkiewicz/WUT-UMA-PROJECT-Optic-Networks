from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest.RandomForest import RandomForest
from sklearn.metrics import accuracy_score
from utils.cross_validation import cross_validation, train_test_split_cv
import pandas as pd

def test_split():
    data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    Xx_train, Xx_test, yy_train, yy_test = train_test_split_cv(data.drop(columns=["OSNR"]), data["OSNR"], 100, 0)

    assert type(X_train) == type(Xx_train)
    assert type(X_train[0]) == type(Xx_train[0])
    assert type(X_test) == type(Xx_test)
    assert type(X_test[0]) == type(Xx_test[0])
    assert type(y_train) == type(yy_train)
    assert type(y_train[0]) == type(yy_train[0])
    assert type(y_test) == type(yy_test)
    assert type(y_test[0]) == type(yy_test[0])
    assert len(X_train[0]) == len(Xx_train[0])
    assert len(X_test[0]) == len(Xx_test[0])
    assert len(y_train[0]) == len(yy_train[0])
    assert len(y_test[0]) == len(yy_test[0])

def test_sample():
    data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)
    Xx_train, Xx_test, yy_train, yy_test = train_test_split_cv(data.drop(columns=["OSNR"]), data["OSNR"], 100, 0)
    model = RandomForest()
    x, y = model._sample(Xx_train, yy_train)
    assert True