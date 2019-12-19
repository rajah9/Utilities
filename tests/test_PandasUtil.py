import pandas as pd
import numpy as np
import unittest
import logging
from pandas.util.testing import assert_frame_equal

from PandasUtil import PandasUtil
from LogitUtil import logit
from ExecUtil import ExecUtil
from FileUtil import FileUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Test_PandasUtil(unittest.TestCase):
    def setUp(self):
        self.pu = PandasUtil()
        self.spreadsheet_name = 'small.xls'
        self.csv_name = 'small.csv'
        self.worksheet_name = 'test'
        self.platform = ExecUtil.which_platform()
        self.list_of_dicts = [
            {'name': 'Rajah', 'number': 42},
            {'name': 'JT', 'number': 2},
            {'name': 'Maddie', 'number': 26},
            {'number': -1},
        ]

    def tearDown(self) -> None:
        fu = FileUtil()
        fu.delete_file(self.spreadsheet_name)
        fu.delete_file(self.csv_name)

    # Return a tiny test dataframe
    def my_test_df(self):
        # Example dataframe from https://www.geeksforgeeks.org/python-pandas-dataframe-dtypes/
        df = pd.DataFrame({'Weight': [45, 88, 56, 15, 71],
                           'Name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
                           'Age': [14, 25, 55, 8, 21]})

        # Create and set the index
        index_ = [0, 1, 2, 3, 4]
        df.index = index_

        return df

    @logit()
    def test_unique_values(self):
        col_name = 'number'
        expected = [x[col_name] for x in self.list_of_dicts]
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        actual = self.pu.unique_values(df, column_name=col_name)
        self.assertListEqual(expected, actual)


    @logit()
    def test_filename(self):
        uninitializedFn = self.pu.filename
        self.assertIsNone(uninitializedFn)
        myFilename = 'xyzzy'
        self.pu.filename = myFilename
        self.assertEqual(myFilename, self.pu.filename)

    @logit()
    def test_worksheetName(self):
        testWorksheet = 'First sheet'
        self.pu.worksheetName = testWorksheet
        self.assertEqual(testWorksheet, self.pu.worksheetName)

    @logit()
    def test_convert_dict_to_dataframe(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        self.assertEqual(len(self.list_of_dicts), len(df))

    @logit()
    def test_without_null_rows(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        col_to_nullify = 'name'
        logger.debug(f'Len of df.name is {len(df[col_to_nullify])}')
        logger.debug(f'Using .count() to count non-nulls in column "{col_to_nullify}" = {df[col_to_nullify].count()}.')
        df_without_null = self.pu.without_null_rows(df=df, column_name=col_to_nullify)
        logger.debug(f'Len of df_without_null is {len(df_without_null)}')
        self.assertEqual(df[col_to_nullify].count(), len(df_without_null))

    @logit()
    def test_get_df_headers(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_entry = self.list_of_dicts[0]
        self.assertListEqual(list(first_entry.keys()), self.pu.get_df_headers(df))

    @logit()
    def test_without_null_rows2(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        col_to_nullify = 'noSuchColumnInTheDf'
        df_without_null = self.pu.without_null_rows(df=df, column_name=col_to_nullify) # Missing col should return empty df.
        assert_frame_equal(PandasUtil.empty_df(), df_without_null)

    @logit()
    def test_write_df_to_excel(self):
        df = self.my_test_df()
        self.pu.write_df_to_excel(df=df, excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        df2 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name, index_col=0)
        logger.debug(f'my test df: {df.head()}')
        logger.debug(f'returned from read_df: {df2.head()}')
        assert_frame_equal(df,df2)
        # Now test that an empty df does not write
        empty_df = PandasUtil.empty_df()
        self.assertFalse(self.pu.write_df_to_excel(df=empty_df))

    @logit()
    def test_write_df_to_csv(self):
        df = self.my_test_df()
        self.pu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=False)
        df2 = self.pu.read_df_from_csv(csv_file_name=self.csv_name, index_col=0)
        assert_frame_equal(df, df2)

    def test_read_df_from_csv(self):
        df = self.my_test_df()
        self.pu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=False)
        df2 = self.pu.read_df_from_csv(csv_file_name=self.csv_name, index_col=0)
        assert_frame_equal(df, df2)

    @logit()
    def test_read_df_from_excel(self):
        df = self.my_test_df()
        self.pu.write_df_to_excel(df=df, excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)
        df2 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)

        assert_frame_equal(df,df2)
        df3 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet='noSuchWorksheet')
        assert_frame_equal(PandasUtil.empty_df(), df3)

    @logit()
    def test_get_rowCount_colCount(self):
        df = self.my_test_df()
        row, col = self.pu.get_rowCount_colCount(df)
        self.assertEqual(row, len(df))
        self.assertEqual(col, len(self.pu.get_df_headers(df)))

    @logit()
    def test_select(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_dict = self.list_of_dicts[0]        # should be {name: 'Rajah', number: 42}
        first_dict_keys = list(first_dict.keys()) # should be ['name', 'number']
        first_key = first_dict_keys[0]            # should be 'name'
        second_key = first_dict_keys[1]           # should be 'number'
        row1 = self.pu.select(df=df, column_name=first_key, match_me=first_dict[first_key]) # select * where name='Rajah'
        el = row1[second_key].item()              # should be 42
        logger.debug(f'el2 is of type {type(el)} and contains {el}')
        self.assertEqual(first_dict[second_key], el)

    def my_f(self, x:int) -> int:
        return x * x

    def my_func(self) -> list:
        df = self.pu.df
        col_of_interest = df['number']
        return [self.my_f(x) for x in col_of_interest]

    def test_add_new_col(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        new_col_name = 'squared'
        df = self.pu.add_new_col(df, new_col_name, self.my_func)
        expected = self.my_func()
        actual = df[new_col_name].tolist()
        self.assertListEqual(actual, expected)

    @logit()
    def test_drop_col(self):
        df = self.my_test_df()
        before_drop = self.pu.get_df_headers(df) # should be ['Name', 'Weight', 'Age']
        # pick one to drop.
        col_to_drop = before_drop[1]
        self.pu.drop_col(df, columns=col_to_drop)
        after_drop = before_drop
        after_drop.remove(col_to_drop)
        self.assertListEqual(self.pu.get_df_headers(df), after_drop)

    @logit()
    def test_set_index(self):
        df = self.my_test_df() # this has an index
        df_new_index = self.pu.set_index(df=df, columns='Name', is_in_place=False)
        after_idx = pd.Index(df_new_index)
        self.assertFalse(after_idx.empty)

    @logit()
    def test_reset_index(self):
        df = self.my_test_df()
        cols_before = self.pu.get_df_headers(df)
        col_to_index_and_reset = 'Age'
        self.pu.set_index(df=df, columns=col_to_index_and_reset, is_in_place=True) # set the Age as the index.
        logger.debug(f'headers are {cols_before}')
        df_no_index = self.pu.reset_index(df, is_in_place=False, is_dropped=True)
        logger.debug(f'no-index headers are {self.pu.get_df_headers(df_no_index)}')
        expected = cols_before
        expected.remove(col_to_index_and_reset)
        self.assertListEqual(expected, self.pu.get_df_headers(df=df_no_index))

    @logit()
    def test_reorder_col(self):
        df = self.my_test_df()
        orig_cols = self.pu.get_df_headers(df) # Should be ['Weight', 'Name', 'Age']
        new_order = [2, 0, 1]
        new_cols = [orig_cols[i] for i in new_order]
        reordered_df = self.pu.reorder_cols(df=df, columns=new_cols)
        actual = self.pu.get_df_headers(reordered_df)
        self.assertListEqual(new_cols, actual)

    @logit()
    def test_replace_col(self):
        df = self.my_test_df()
        orig_weights = list(df.Weight)
        expect_weights = [x * 2 for x in orig_weights]
        replace_dict = dict(zip(orig_weights, expect_weights))
        df2 = self.pu.replace_col(df, 'Weight', replace_dict)
        self.assertListEqual(expect_weights, list(df2.Weight))

    @logit()
    def test_replace_col_err(self):
        df = self.my_test_df()
        orig_weights = list(df.Weight)
        expect_weights = [x * 2 for x in orig_weights]
        replace_dict = dict(zip(orig_weights, expect_weights))
        bad_column = 'noSuchColumn'
        with self.assertLogs(PandasUtil.__name__, level='DEBUG') as cm:
            df2 = self.pu.replace_col(df, bad_column, replace_dict)
            assert_frame_equal(PandasUtil.empty_df(), df2)
            self.assertTrue(next((True for line in cm.output if bad_column in line), False))
    @logit()
    def test_replace_col_using_func(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        col_name = 'number'
        old_list = df[col_name]
        expected = [self.my_f(x) for x in old_list]
        df2 = self.pu.replace_col_using_func(df, col_name, self.my_f)
        actual = df2[col_name].tolist()
        self.assertListEqual(actual, expected)

    @logit()
    def test_coerce_to_string(self):
        df = self.my_test_df()
        age_series = df['Age']
        df2 = self.pu.coerce_to_string(df, 'Age')
        age_str_series = df2['Age']
        expected = [str(age) for age in age_series]
        actual = list(age_str_series)
        self.assertListEqual(expected, actual)

    @logit()
    def test_coerce_to_numeric(self):
        df = self.my_test_df()
        age_series = df['Age'].astype(str).astype(int)
        df2 = self.pu.coerce_to_numeric(df, 'Age')
        age_num_series = df2['Age']
        expected = age_series.tolist()
        actual = list(age_num_series)
        self.assertListEqual(expected, actual)
        weight_series = df['Weight'].astype(str).astype(int)
        df3 = self.pu.coerce_to_numeric(df, ['Age', 'Weight'])
        weight_num_series = df3['Weight']
        self.assertListEqual(weight_series.tolist(), weight_num_series.tolist())

    def is_adult(self, age:list):
        return age >= 21

    @logit()
    def test_mark_rows(self):
        df = self.my_test_df()
        mark = self.pu.mark_rows(df, 'Age', self.is_adult)
        logger.debug(f'marked rows are: {mark}')
        ans = [x >= 21 for x in df['Age']]
        self.assertListEqual(ans, mark.tolist())

    @logit()
    def test_is_empty(self):
        df = self.my_test_df()
        self.assertFalse(self.pu.is_empty(df))
        df2 = self.pu.empty_df()
        self.assertTrue(self.pu.is_empty(df2))


from PandasUtil import DataFrameSplit

class Test_DataFrameSplit(unittest.TestCase):
    @logit()
    def test_iterators(self):
        d = {'num': [1,2,3,4,5,6]}
        df = pd.DataFrame(data=d)
        little_dfs = DataFrameSplit(my_df=df, interval=4)
        combined_sizes = 0
        for i, little_df in enumerate(little_dfs):
            logger.debug(f'Set {i}: {little_df.head()}')
            combined_sizes += len(little_df)
        self.assertEqual(len(df), combined_sizes)