from category_encoders.target_encoder import TargetEncoder

class TargetEncoderCustom:
    """
        Class to encode the data in each column
        A target value for each feature is selected and the probability of that instance is calculated
        These probabilities replace the values in the feature columns
    """

    def __init__(self):
        self.encoder = TargetEncoder()
        self.column_encoder_dict = {}

    def fit(self, X, y):
        """
        Fit the target encoder

        Args:
            X: Data to encode
            y: Features
        Returns:
            x_copy: Encoded copy of the data with target probabilities
        """

        columns_ls = X.columns
        X[columns_ls] = X[columns_ls].astype(str)

        for column in columns_ls:
            curr_encoder = TargetEncoder(smoothing=5, min_samples_leaf=50)
            curr_encoder.fit(X[column], y)
            self.column_encoder_dict[column] = curr_encoder

        return self

    def transform(self, X, y=None):
        """
        Create a copy of the data and run it through the encoder

        Args:
            X: Data to encode
            y: Features (not used)
        Returns:
            x_copy: Encoded copy of the data with target probabilities
        """

        x_copy = X.copy()
        new_ls = x_copy.columns

        x_copy.loc[:, new_ls] = x_copy.loc[:, new_ls].astype(str)
        for column in new_ls:
            x_copy.loc[:, column] = self.column_encoder_dict[column].transform(
                x_copy[column])

        return x_copy
