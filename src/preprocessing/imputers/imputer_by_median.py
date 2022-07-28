
class ImputerByMedian:

    def __init__(self):
        self.medians_dict = {}

    def fit(self, X, y=None):

        for col in X.columns:  # todo to var X[self.num_features_ls].columns
            median = X[col].median()
            self.medians_dict.update({col: median})

        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        new_ls = X_.columns
        for col in new_ls:
            X_.loc[:, col] = X_[col].fillna(self.medians_dict[col])

        return X_
