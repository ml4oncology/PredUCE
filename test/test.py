import unittest

import pandas as pd

from .util import test_date_is_ordered

class Tester(unittest.TestCase):
    """NOTE: This is just for proof-of-concept, you can scrap these tests if you want
    """
    def test_ordering(self):
        df = pd.read_parquet('./data/treatment_centered_clinical_dataset.parquet.gzip')
        test_date_is_ordered(df, date_col='treatment_date', patient_col='mrn')
        df = pd.read_parquet('./data/raw/opis.parquet.gzip')
        test_date_is_ordered(df, date_col='Trt_Date', patient_col='Hosp_Chart')
        df = pd.read_parquet('./data/external/symptom.parquet.gzip')
        test_date_is_ordered(df, date_col='survey_date', patient_col='mrn')

if __name__ == '__main__':
    """
    > python -m unittest test/test.py
    Reference: docs.python.org/3/library/unittest.html
    """
    unittest.main()