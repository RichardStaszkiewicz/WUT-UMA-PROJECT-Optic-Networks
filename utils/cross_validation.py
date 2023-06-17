import pandas as pd
import numpy as np

def train_test_split_cv(data, target, step, i):
    test_data = data.iloc[step * i : step * (i + 1)]
    learn_data = pd.concat(
        [
            data.iloc[0 : step * i],
            data.iloc[step * (i + 1) : -1],
        ]
    )
    test_target = target.iloc[step * i : step * (i + 1)]
    learn_target = pd.concat(
        [
            target.iloc[0 : step * i],
            target.iloc[step * (i + 1) : -1],
        ]
    )

    learn_target = np.swapaxes(np.array([learn_target.values]), 0, 1)
    test_target = np.swapaxes(np.array([test_target.values]), 0, 1)

    return learn_data.values, test_data.values, learn_target, test_target

def cross_validation(model, data, target, cv, validation_fc, evaluation_fc=[np.mean, np.std], verbose=False):
    step = len(data) // cv
    results = []
    for i in range(cv):
        X_train, X_test, y_train, y_test = train_test_split_cv(data, target, step, i)
        print(f"{i}: Splitted. X = {X_train.shape}") if verbose else None
        model.fit(X_train, y_train)
        print(f"{i}: Trained") if verbose else None
        preds = model.predict(X_test)
        results.append(validation_fc(y_test, preds))
        print(f"result {i}: {results[-1]}") if verbose else None
    return [e(results) for e in evaluation_fc]