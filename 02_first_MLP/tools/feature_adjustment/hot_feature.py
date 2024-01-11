# OBSOLETO
# Practico si se quisiera optimizar la categorización por multihilos columna por columna
# inutil para la presentacion del sprint 2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# create a numpy array from list
# arr = np.array([0.0],)


def one_hot_train(df: pd.DataFrame) -> tuple:
    # OBSOLETA
    # OrdinalEncoder está siendo utilizado columna por columna en vez de todo el dataframe
    # La version original utilizaba LabelEncoder, y previo a la implementacion del
    # .transform() se operaba manualmente con los registros que no formaban parte del train set.
    """One hot encoding for train dataset
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to encode with categorical features
    Returns
    -------
    list
        Element 1: pd.DataFrame with encoded features
        Element 2: dict with fitted models
    """
    # LabelEncoder training
    fit_label_enc = [LabelEncoder().fit(df[col]) for col in df.columns]
    # fit_label_enc = [OrdinalEncoder().fit(df[[col]]) for col in df.columns]
    label_trsf = [enc.transform(df[col]) for enc, col in zip(fit_label_enc, df.columns)]

    # OneHotEncoder training
    fit_onehot_enc = [
        OneHotEncoder(sparse_output=False).fit(label_df.reshape(len(label_df), 1))
        for label_df in label_trsf
    ]
    onehot_trsf = [
        enc.transform(label_df.reshape(len(label_df), 1))
        for enc, label_df in zip(fit_onehot_enc, label_trsf)
    ]

    # Feature adjust
    feature_names = [
        enc.get_feature_names_out(input_features=[col])
        for enc, col in zip(fit_onehot_enc, df.columns)
    ]
    for ix, elem in enumerate(zip(feature_names, fit_label_enc, df.columns)):
        lista_feature, modelo, col_labels = elem
        feature_list = [
            modelo.inverse_transform([int(label.split("_")[-1])]).tolist()[0]
            for label in lista_feature
        ]
        # feature_list = [modelo.inverse_transform(np.array([float(label.split("_")[-1])])).reshape(1, -1) for label in lista_feature]
        feature_names[ix] = [f"{col_labels}_{feature}" for feature in feature_list]

    # Model Assembly
    models = dict(zip(df.columns, list(zip(fit_label_enc, fit_onehot_enc))))

    # Result Assembly
    result = pd.DataFrame()
    for one_hot, feature in zip(onehot_trsf, feature_names):
        result = pd.concat([result, pd.DataFrame(one_hot, columns=feature)], axis=1)

    return result, models


def one_hot_transformation(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """One hot encoding for test dataset
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to encode with categorical features
    models : dict
        Dictionary with fitted models
    Returns
    -------
    pd.DataFrame
        Dataframe with encoded features
    """
    result = pd.DataFrame()
    for col in df.columns:
        label_enc, one_hot_enc = models[col]

        # check if data structure from training, fits the process
        unique_set = df[col].drop_duplicates()
        if not unique_set.isin(models[col][0].classes_).all():
            excluded = df[col][~df[col].isin(models[col][0].classes_)]
            print("Warning: some values in the test set are not in the training set")
            print(f"Excluded values counts: {excluded.value_counts()}")
            print(
                "Replacing excluded values with the most frequent value in the training set"
            )
            excluded = excluded.drop_duplicates().to_list()
            safety_value = df[col].value_counts().index[0]
            safety_value = [safety_value for _ in range(len(excluded))]
            replacing_dict = dict(zip(excluded, safety_value))
            print("\n", str(replacing_dict), "\n")
            replacing_dict |= dict(
                zip(models[col][0].classes_, models[col][0].classes_)
            )
            df[col] = df[col].map(replacing_dict)
            print("NaNs in feature values after replacement: ", df[col].isna().sum())

        try:
            label_trsf = label_enc.transform(df[col].dropna())
            if len(df[col]) != len(df[col].dropna()):
                print(f"Warning: NaNs in feature [{col}]: ", df[col].isna().sum())
                # TODO: keep tracking

        except ValueError as e:
            print(f"Error: {e}")
            print(
                "Developer notes:\n\t- DONE: Check if data structure from training, fits the process"
            )
            raise ValueError("Some values in the test set are not in the training set")

        one_hot_trsf = one_hot_enc.transform(label_trsf.reshape(len(label_trsf), 1))
        feature_names = one_hot_enc.get_feature_names_out(input_features=[col])
        feature_names = [
            label_enc.inverse_transform([int(label.split("_")[-1])]).tolist()[0]
            for label in feature_names
        ]
        feature_names = [f"{col}_{feature}" for feature in feature_names]
        result = pd.concat(
            [result, pd.DataFrame(one_hot_trsf, columns=feature_names)], axis=1
        )
    return result


if __name__ == "__main__":
    # BORRAR
    from typing import Tuple
    from sklearn.preprocessing import OrdinalEncoder

    def preprocess_data(
        train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pre processes data for modeling. Receives train, val and test dataframes
        and returns numpy ndarrays of cleaned up dataframes with feature engineering
        already performed.

        Arguments:
            train_df : pd.DataFrame
            val_df : pd.DataFrame
            test_df : pd.DataFrame

        Returns:
            train : np.ndarrary
            val : np.ndarrary
            test : np.ndarrary
        """
        # Print shape of input data
        print("Input train data shape: ", train_df.shape)
        print("Input val data shape: ", val_df.shape)
        print("Input test data shape: ", test_df.shape, "\n")

        # Make a copy of the dataframes
        working_train_df = train_df.copy()
        working_val_df = val_df.copy()
        # TODO: Borrar parche:
        # test_df.loc[test_df["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = test_df["NAME_FAMILY_STATUS"].value_counts().index.to_list()[0]
        working_test_df = test_df.copy()

        # 1. Correct outliers/anomalous values in numerical
        # columns (`DAYS_EMPLOYED` column).
        working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
        working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
        working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

        # 2. TODO Encode string categorical features (dytpe `object`):
        amount_of_categories = (
            working_train_df.loc[:, (working_train_df.dtypes == "object").values]
            .apply(lambda x: x.to_frame().drop_duplicates().value_counts(), axis=0)
            .sum()
        )

        # Two unique categories treatment
        two_categories_features = amount_of_categories[
            amount_of_categories == 2
        ].index.to_list()

        encoder_models = [
            OrdinalEncoder().fit(working_train_df[[sample]])
            for sample in two_categories_features
        ]
        encoder_zip = tuple(zip(encoder_models, two_categories_features))
        encoded_values = [
            model.transform(working_train_df[[sample]]) for model, sample in encoder_zip
        ]
        working_train_df.loc[:, two_categories_features] = encoded_values
        encoded_values = [
            model.transform(working_val_df[[sample]]) for model, sample in encoder_zip
        ]
        working_val_df.loc[:, two_categories_features] = encoded_values
        encoded_values = [
            model.transform(working_test_df[[sample]]) for model, sample in encoder_zip
        ]
        working_test_df.loc[:, two_categories_features] = encoded_values

        # Rest of the Categories treatment
        plus_two_categories_features = amount_of_categories[
            amount_of_categories > 2
        ].index.to_list()

        # Train set
        encoded_train_df, models = one_hot_train(
            working_train_df[plus_two_categories_features]
        )
        concat_list = [
            working_train_df.drop(columns=plus_two_categories_features).reset_index()
        ] + [
            encoded_train_df
        ]  # NOTE: LabelEncoder no genera columna '_nan', por eso el reset_index(drop=True) no está activado
        working_train_df = pd.concat(concat_list, axis=1)

        # Validation set
        encoded_val_df = one_hot_transformation(
            working_val_df[plus_two_categories_features], models
        )
        concat_list = [
            working_val_df.drop(columns=plus_two_categories_features).reset_index()
        ] + [
            encoded_val_df
        ]  # NOTE: LabelEncoder no genera columna '_nan', por eso el reset_index(drop=True) no está activado
        working_val_df = pd.concat(concat_list, axis=1)

        # Test set
        encoded_test_df = one_hot_transformation(
            working_test_df[plus_two_categories_features], models
        )
        concat_list = [
            working_test_df.drop(columns=plus_two_categories_features).reset_index()
        ] + [
            encoded_test_df
        ]  # NOTE: LabelEncoder no genera columna '_nan', por eso el reset_index(drop=True) no está activado
        working_test_df = pd.concat(concat_list, axis=1)

        # 3. TODO Impute values for all columns with missing data or, just all the columns.
        # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
        # Again, take into account that:
        #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
        #     working_test_df).
        #   - In order to prevent overfitting and avoid Data Leakage you must use only
        #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
        #     model to transform all the datasets.

        # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
        # Please use sklearn.preprocessing.MinMaxScaler().
        # Again, take into account that:
        #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
        #     working_test_df).
        #   - In order to prevent overfitting and avoid Data Leakage you must use only
        #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
        #     model to transform all the datasets.

        return working_train_df, working_val_df, working_test_df

    X_train, X_val, X_test = None, None, None
    train_data, val_data, test_data = preprocess_data(X_train, X_val, X_test)
