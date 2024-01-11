import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from tools.feed import data_info


def create_sparse_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    # Crear una instancia de OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    # Ajustar y transformar los datos
    category_sentiment_encoded = encoder.fit_transform(df[[feature]])
    # Convertir a DataFrame y unirlo al DataFrame original
    sentiment_encoded = pd.DataFrame(
        category_sentiment_encoded,
        columns=encoder.get_feature_names_out([feature]),
        index=df.index,
    )

    return sentiment_encoded


def resample_dataset_s(df, freq, period=None):
    import sys

    print(
        "ERROR: que el dtype sea distinto de float significa muchas otras cosas, pero no significa que sean 'no numéricas'."
    )
    sys.exit(0)
    non_numeric_column = (
        data_info(df).loc[(data_info(df)["dtype"] != float).values].columna.values
    )

    df_categorical = df[non_numeric_column]
    df_numerical = df.drop(non_numeric_column, axis=1)

    df_categorical = df_categorical.resample(freq).agg(lambda x: list(x))
    df_numerical = df_numerical.resample(freq).sum()

    if period is not None and period == "12H":
        # time delta last_date - 12 hours
        last_date = df_numerical.index[-1]
        last_date = last_date - pd.Timedelta(hours=12)

        df_categorical = df_categorical.loc[last_date:]
        df_numerical = df_numerical.loc[last_date:]

    elif period != None and period != "12H":
        print("period must be None or '12H'")

    return pd.concat([df_categorical, df_numerical], axis=1)


def resample_dataset(df, freq):
    non_numeric_column = (
        data_info(df).loc[(data_info(df)["dtype"] != float).values].columna.values
    )

    df_categorical = df[non_numeric_column]
    df_numerical = df.drop(non_numeric_column, axis=1)

    df_categorical = df_categorical.resample(freq).agg(lambda x: list(x))
    df_numerical = df_numerical.resample(freq).mean()

    return pd.concat([df_categorical, df_numerical], axis=1)
