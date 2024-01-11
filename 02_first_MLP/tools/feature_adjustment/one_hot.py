# En desuso debido a que solo devuelve las columnas de la matriz decodificada
# Y no se tiene acceso al modelo entrenado para poder transformar nuevos datos

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def obtain_models():
    return LabelEncoder(), OneHotEncoder(sparse=False)


def fit_transform_reshape(
    df: pd.DataFrame, target: str, label_enc: LabelEncoder, onehot_enc: OneHotEncoder
):
    # TODO: USE A FUNCTION THAT RETURN THE FITTED INSTANCE OF THE MODEL
    label_enc = label_enc.fit(df[target])
    label_pred = label_enc.transform(df[target])
    label_pred = label_pred.reshape(len(label_pred), 1)
    onehot_values = onehot_enc.fit_transform(label_pred)
    return label_enc, onehot_values


def dataframe_encoded_features(
    target: str, label_enc: LabelEncoder, onehot_enc: OneHotEncoder
) -> list:
    feature_names = onehot_enc.get_feature_names_out(input_features=[target])
    feature_names = [
        label_enc.inverse_transform([int(elem.split("_")[-1])]).tolist()[0]
        for elem in feature_names
    ]
    return [f"{target}_" + str(ft_names) for ft_names in feature_names]


def dataframe_encoded(one_hot_values, feature_names: list) -> pd.DataFrame:
    return pd.DataFrame(one_hot_values, columns=feature_names)


def one_hot_feature(df: pd.DataFrame, target: str) -> pd.DataFrame:
    "Feature 'target' must be categorical."
    label_enc, onehot_enc = obtain_models()
    label_enc, onehot_values = fit_transform_reshape(df, target, label_enc, onehot_enc)
    feature_names = dataframe_encoded_features(target, label_enc, onehot_enc)
    return dataframe_encoded(onehot_values, feature_names)
