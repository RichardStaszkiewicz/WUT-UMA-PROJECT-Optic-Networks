import pandas as pd
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

def preprocess(all_data, target_name):
    target = all_data[target_name]
    data = all_data.drop(columns=[target_name])

    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)

    numerical_columns = numerical_columns_selector(data)
    categorical_columns = categorical_columns_selector(data)

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns),
    ]
    )
    return data, target, preprocessor


if __name__ == "__main__":
    categorical = ["transponder_modulation", "transponder_bitrate"]
    discard = ["Unnamed: 0"]
    all_data = pd.read_csv("./data/data_janos-us-ca.xml.csv")
    for d in discard:
        if d in all_data.columns:
            all_data = all_data.drop(columns=d)
    all_data[categorical] = all_data[categorical].astype(str)
    data, target, preprocessor = preprocess(all_data.sample(100, random_state=42), "OSNR")
    model = make_pipeline(preprocessor, LinearRegression())
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, random_state=42
    )
    _ = model.fit(data_train, target_train)
    cv_results = cross_validate(model, data, target, cv=5)
    scores = cv_results["test_score"]
    print(
        "The mean cross-validation accuracy is: "
        f"{scores.mean():.3f} ± {scores.std():.3f}"
    )
    # print(model.score(data_test, target_test))

    # print(model)
    # print([type(x) for x in all_data.iloc[1]])
