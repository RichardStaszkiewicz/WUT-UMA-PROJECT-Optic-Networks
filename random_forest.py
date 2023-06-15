from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from random_forest.RandomForest import RandomForest
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = RandomForest()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))