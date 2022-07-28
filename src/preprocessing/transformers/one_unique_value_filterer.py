from pandas import DataFrame


class OneUniqueValueFilterer:
    """
    Class to filter all columns have more than 1 unique
    value.
    """

    COL_AX = 'columns'

    def __init__(self):
        self.features_to_drop_list = list()

    def fit(self, df, y=None):
        """
        Adding columns that has more than 1 unique value
        to self.feature_list.

        Args:
            df (DataFrame): data table to filter columns.

        Returns:
            OneUniqueValue.

        """
        df = df.loc[:, ~df.columns.duplicated()]
        for col in df.columns:
            temp_col = df[col]
            try:
                temp_col = df[col].astype(str)
            except:
                pass
            if temp_col.nunique(dropna=False) == 1:
                self.features_to_drop_list.append(col)

        return self

    def transform(self, df):
        """
        Reassigning dataframe to the new with corresponding
         columns from self.feature_list.

        Args:
            df (DataFrame): data table to choose columns from.

        Returns:
            DataFrame. Contains only columns with more than
                one unique value.

        """
        df_ = df.copy()

        columns_ls = list(filter(lambda x: x in set(self.features_to_drop_list), list(df.columns)))
        df_ = df_.drop(columns=columns_ls, axis=self.COL_AX)

        return df_
