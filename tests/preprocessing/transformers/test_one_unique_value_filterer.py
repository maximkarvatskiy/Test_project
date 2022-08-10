from src.preprocessing.transformers.one_unique_value_filterer import OneUniqueValueFilterer
from tests.preprocessing.data.transformers import one_unique_value_filterer_datasets as datasets
import collections

class TestOneUniqueValueFilterer:
    """
    Test the OneUniqueValueFilterer on various datasets to ensure it properly identifies and drops
    features with only one unique value in various circumstances
    Use the datasets created in one_unique_value_filterer_datasets
    Assert that the filter worked correctly to check results
    """

    def fit_transform(self, df):
        """
        Fit the 1 unique value filterer to the data
        Args:
            df (DataFrame): Data table to filter
        Returns:
            df (DataFrame): Original data table
            res: Filtered data table
        """
        filterer = OneUniqueValueFilterer()
        filterer.fit(df)
        res = filterer.transform(df)

        return df, res

    def test_int_column_with_all_uniq_result_drop_column(self):
        """
        Calls the fit_transform method on the int_column_with_one_uniq dataset
        Checks which columns are still in the dataframe and if any rows were dropped
        Asserts if the data was transformed correctly or not
        If not, this function raises an assertion error
        """
        df, res = self.fit_transform(datasets.int_column_with_one_uniq)
        assert list(res.columns) == ["Age", "Country"]
        assert len(df) == len(res)

    def test_string_column_with_all_uniq_result_drop_column(self):
        """
        Calls the fit_transform method on the string_column_with_one_uniq dataset
        Checks which columns are still in the dataframe and if any rows were dropped
        Asserts if the data was transformed correctly or not
        If not, this function raises an assertion error
        """
        df, res = self.fit_transform(datasets.string_column_with_one_uniq)
        assert list(res.columns) == ["id", "Age"]
        assert len(df) == len(res)

    def test_int_column_with_all_uniq_and_missing_result_not_drop_column(self):
        """
        Calls the fit_transform method on the int_column_with_one_uniq_and_missing dataset
        Checks which columns are still in the dataframe and if any rows were dropped
        Asserts if the data was transformed correctly or not
        If not, this function raises an assertion error
        """
        df, res = self.fit_transform(datasets.int_column_with_one_uniq_and_missing)
        assert list(res.columns) == ["id", "Age", "Country"]
        assert len(df) == len(res)

    def test_float_column_with_all_missing_result_drop_column(self):
        """
        Calls the fit_transform method on the float_column_with_one_missing dataset
        Checks which columns are still in the dataframe and if any rows were dropped
        Asserts if the data was transformed correctly or not
        If not, this function raises an assertion error
        """
        df, res = self.fit_transform(datasets.float_column_with_one_missing)
        assert collections.Counter(res.columns) == collections.Counter(["id", "Country"])
        assert len(df) == len(res)

"""
Call all functions in the TestOneUniqueValueFilterer class
If this code does not raise an assertion error then the OneUniqueValueFilterer is working correctly
"""
uniq = TestOneUniqueValueFilterer()
uniq.test_int_column_with_all_uniq_result_drop_column()
uniq.test_string_column_with_all_uniq_result_drop_column()
uniq.test_int_column_with_all_uniq_and_missing_result_not_drop_column()
uniq.test_float_column_with_all_missing_result_drop_column()


