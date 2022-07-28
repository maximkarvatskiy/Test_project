import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerCustom:

    def __init__(self):
        self.columns_scaler_dict = {}

    def fit(self, X, y=None):

        for column in X.columns:
            column_values_array = np.array(X[column]).reshape(-1, 1)
            scaler = MinMaxScaler().fit(column_values_array)
            self.columns_scaler_dict[column] = scaler

        return self

    def transform(self, X, y=None):
        X_copy = pd.DataFrame.copy(X)
        for column in X_copy.columns:
            column_values_array = np.array(X_copy[column]).reshape(-1, 1)
            X_copy[column] = self.columns_scaler_dict[column].transform(
                column_values_array)

        return X_copy

    def inverse_transform(self, X, y=None):
        X_copy = pd.DataFrame.copy(X)
        for column in X_copy.columns:  # TODO: 'if' statement is not necessary here (only temporary fix)
            if column in self.columns_scaler_dict.keys():
                column_values_array = np.array(X_copy[column]).reshape(-1, 1)
                X_copy[column] = self.columns_scaler_dict[column].inverse_transform(column_values_array)
        return X_copy

