import pandas as pd
import numpy as np

"""
Create 4 datasets to test the one unique value filter
Use variables of various types (int, string, float) to ensure the filter can detect them
Use datasets with missing values to ensure they are detected by the filter
If a column has 1 unique value and missing values, it should not be dropped
If a column has only missing values, it should be dropped
"""

int_column_with_one_uniq = pd.DataFrame(
    data={"id": [1, 1, 1, 1, 1],
          "Age": [1, 1, 22, 33, 95],
          "Country": ['Japan', 'Indonesia', 'India', 'Philippines', 'Brazil']}
)
"""
The "id" column should be dropped
"""

string_column_with_one_uniq = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [1, 1, 22, 33, 95],
          "Country": ['Japan', 'Japan', 'Japan', 'Japan', 'Japan']}
)
"""
The "Country" column should be dropped
"""

int_column_with_one_uniq_and_missing = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [1, 1, 1, 1, np.nan],
          "Country": ['Georgia', 'Ukraine', 'China', 'Belarus', 'Japan']}
)
"""
No column should be dropped
Make sure the "Age" column is not dropped
"""

float_column_with_one_missing = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [np.nan, np.nan, np.nan, np.nan, np.nan],
          "Country": ['Georgia', 'Ukraine', 'China', 'Belarus', 'Japan']}
)
"""
The "Country" column should be dropped
"""