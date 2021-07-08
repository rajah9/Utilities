import logging
import unittest
from datetime import datetime
from math import sqrt, isnan, nan
from typing import List

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from CollectionUtil import CollectionUtil
from DateUtil import DateUtil
from ExecUtil import ExecUtil
from FileUtil import FileUtil
from LogitUtil import logit
from PandasUtil import PandasUtil, PandasDateUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

Dates = List[datetime]

"""
Interesting Python features:
* Does some dict comprehension in test_replace_col_names. 
** Uses self.assertLogs to ensure that an error message is logged in the calling routine.
* Uses next to figure out if it found expected_log_message in cm.output:
** self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
* test_aggregates uses some numpy aggregates, such as sum, min, max, and mean.
* in test_select_blanks, found the first element in a Series with .iloc[0]
* in test_select_blanks, set an element within a dataframe using df.at
* in test_mark_isnull, used an iloc[-1] to get the last record
* test_replace_col_using_mult_cols uses an inner method to distinguish calculation of the expected value
* test_select_non_blanks deletes rows based on criteria.
* in test_join_dfs_by_row, uses `act_names.str.contains(new_guy, regex=False).any()` to test if new_guy is anywhere in act_names.
* in test_sma, tests for NaN. 
"""

class Test_PandasUtil(unittest.TestCase):
    def setUp(self):
        self.pu = PandasUtil()
        self.cu = CollectionUtil()
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
                           'Sex' : ['male', 'female', 'male', 'female', 'male'],
                           'Age': [14, 25, 55, 8, 21]})

        # Create and set the index
        index_ = [0, 1, 2, 3, 4]
        df.index = index_
        return df

    @logit()
    def test_pandas_verson(self):
        major, minor, sub = self.pu.pandas_version()
        logger.debug(f'You are running Pandas version {major}.{minor}.{sub}')
        self.assertTrue(((major>0) and (minor>=1) or ((major==0) and (minor>=21))))

    @logit()
    def test_unique_values(self):
        col_name = 'number'
        expected = [x[col_name] for x in self.list_of_dicts]
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        actual = self.pu.unique_values(df, column_name=col_name)
        self.assertListEqual(expected, actual)

    @logit()
    def test_count_by_column(self):
        col_name = 'age'
        def small_df() -> pd.DataFrame:
            d = [
                {'name': 'Alice', 'flip': 'T', 'age': 13},
                {'name': 'Bill', 'flip': 'H', 'age': 13},
                {'name': 'Cory', 'flip': 'H', 'age': 15},
            ]
            return self.pu.convert_dict_to_dataframe(list_of_dicts=d)
        df = small_df()
        unique, counts = np.unique(df[col_name], return_counts=True)
        zip_obj = zip(unique, counts)
        expected = list(zip_obj)[:] # should be: [(13, 2), (15, 1)]
        a = self.pu.count_by_column(df, column_name=col_name)
        actual = list(zip(a.index, a))
        self.assertListEqual(expected, actual)

    @logit()
    def test_join_two_dfs_on_index(self):
        df1 = self.my_test_df()

        def my_test_df2():
            df = pd.DataFrame({'Shoe_size': [4.0, 5.5, 5.5, 10.5, 8.0]})
            # Create and set the index
            index_ = [0, 1, 2, 3, 4]
            df.index = index_
            return df
        df2 = my_test_df2()
        actual = self.pu.join_two_dfs_on_index(df1, df2)
        for index, row in actual.iterrows():
            self.assertEqual(row["Shoe_size"], df2.iloc[index]["Shoe_size"])
            self.assertEqual(row["Name"], df1.iloc[index]["Name"])

    @logit()
    def test_join_dfs_by_column(self):
        # Test 1
        df1 = self.my_test_df()
        df2 = pd.DataFrame({'Shoe_size': [4.0, 5.5, 5.5, 10.5, 8.0]})
        actual = self.pu.join_dfs_by_column([df1, df2])
        for index, row in actual.iterrows():
            self.assertEqual(row["Shoe_size"], df2.iloc[index]["Shoe_size"])
            self.assertEqual(row["Name"], df1.iloc[index]["Name"])
        # Test 2, three dfs
        df3 = pd.DataFrame({'IQ': [100, 105, 85, 125, 98]})
        actual = self.pu.join_dfs_by_column([df1, df3, df2])
        for index, row in actual.iterrows():
            self.assertEqual(row["Shoe_size"], df2.iloc[index]["Shoe_size"])
            self.assertEqual(row["IQ"], df3.iloc[index]["IQ"])


    @logit()
    def test_join_dfs_by_row(self):
        # Test 1
        df1 = self.my_test_df()
        new_guy = 'Bill'
        new_gal = 'Julie'
        df2 =  pd.DataFrame({'Weight': [110, 115],
                           'Name': [new_guy, new_gal],
                           'Sex' : ['male', 'female'],
                           'Age': [12, 13]})
        exp_len = len(df1) + len(df2)
        actual_df = self.pu.join_dfs_by_row([df1, df2])
        self.assertEqual(exp_len, len(actual_df))
        act_names = actual_df['Name']
        self.assertTrue(act_names.str.contains(new_guy, regex=False).any())
        self.assertTrue(act_names.str.contains(new_gal, regex=False).any())

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
    def test_convert_list_to_dataframe(self):
        # Test 1, with column names
        expected_df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        self.pu.drop_index(expected_df, True)

        def list_of_dicts_to_lists(df: pd.DataFrame) -> list:
            ans = []
            for row in df.itertuples(index=False):
                data = list(row)
                ans.append(data)  # Append because I'm creating a list of lists.
            return ans

        lists = list_of_dicts_to_lists(expected_df)
        actual_df = self.pu.convert_list_to_dataframe(lists=lists, column_names=self.pu.get_df_headers(expected_df))
        # self.pu.set_index(df=actual_df, columns=[0])
        assert_frame_equal(expected_df, actual_df)
        # Test 2, with default column names col00, col01...
        self.pu.replace_col_names_by_pattern(expected_df, prefix='col', is_in_place=True)
        actual_df = self.pu.convert_list_to_dataframe(lists=lists, column_names=None)
        assert_frame_equal(expected_df, actual_df)

    def test_coerece_to_string(self):
        df = self.my_test_df()
        age_series = df['Age'].astype(str)
        self.pu.coerce_to_string(df, 'Age')
        age_num_series = df['Age']
        expected = age_series.tolist()
        actual = list(age_num_series)
        self.assertListEqual(expected, actual, "Fail test 1")

    def test_coerce_to_int(self):
        sizes = [4.0, 5.5, 5.5, 10.5, 8.0]
        col_name = 'Shoe_size'
        df = pd.DataFrame({col_name: sizes})
        exp = [int(x) for x in sizes]
        self.pu.coerece_to_int(df, col_name)
        act = list(df[col_name])
        self.assertListEqual(exp, act)

    def test_convert_matrix_to_dataframe(self):
        exp_df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        act_df = self.pu.convert_matrix_to_dataframe(self.list_of_dicts)
        assert_frame_equal(exp_df, act_df)

    @logit()
    def test_convert_dataframe_to_matrix(self):
        df_orig = self.my_test_df()
        df = self.pu.drop_col_keeping(df_orig, cols_to_keep=['Weight', 'Age'], is_in_place=False)
        expected = df.to_numpy()
        actual = self.pu.convert_dataframe_to_matrix(df)
        for exp, act in zip(expected, actual):
            self.assertListEqual(list(exp), list(act), "Failure test 1")

    @logit()
    def test_convert_dataframe_to_vector(self):
        df_orig = self.my_test_df()
        df = self.pu.drop_col_keeping(df_orig, cols_to_keep='Weight', is_in_place=False)
        expected = df.to_numpy().reshape(-1,)
        actual = self.pu.convert_dataframe_to_vector(df)
        self.assertListEqual(list(expected), list(actual), "Failure test 1")

    @logit()
    def test_convert_dataframe_col_to_list(self):
        df = self.my_test_df()
        col_to_check = 'Age'
        expected = df[col_to_check]
        self.assertListEqual(list(expected), self.pu.convert_dataframe_col_to_list(df, col_to_check))

    @logit()
    def test_sort(self):
        df_orig = self.my_test_df()
        df = self.pu.drop_col_keeping(df_orig, cols_to_keep='Weight', is_in_place=False)
        expected = df.to_numpy().reshape(-1,)
        expected.sort()
        actual = self.pu.sort(df, columns = 'Weight', is_in_place=False, is_asc=True)
        self.assertListEqual(list(expected), list(actual['Weight']), "Failure test 1")

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
        # Test 1
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_entry = self.list_of_dicts[0]
        self.assertListEqual(list(first_entry.keys()), self.pu.get_df_headers(df))
        # Test 2
        df2 = self.pu.empty_df()
        self.assertIsNone(self.pu.get_df_headers(df2))

    @logit()
    def test_set_df_headers(self):
        # Test 1
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_entry = self.list_of_dicts[0]
        new_header = [f'Column{i}' for i in range(len(first_entry))]
        self.pu.set_df_headers(df=df, new_headers=new_header)
        self.assertListEqual(self.pu.get_df_headers(df), new_header)

    @logit()
    def test_without_null_rows2(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        col_to_nullify = 'noSuchColumnInTheDf'
        df_without_null = self.pu.without_null_rows(df=df, column_name=col_to_nullify) # Missing col should return empty df.
        assert_frame_equal(PandasUtil.empty_df(), df_without_null)

    @logit()
    def test_write_df_to_excel(self):
        df = self.my_test_df()
        self.pu.write_df_to_excel(df=df, excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True)
        df2 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name, index_col=0)
        logger.debug(f'my test df: {df.head()}')
        logger.debug(f'returned from read_df: {df2.head()}')
        assert_frame_equal(df,df2)
        # Test 2. Now test that an empty df does not write
        empty_df = PandasUtil.empty_df()
        self.assertFalse(self.pu.write_df_to_excel(df=empty_df))

    @logit()
    def test_write_df_to_csv(self):
        df = self.my_test_df()
        self.pu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=False)
        df2 = self.pu.read_df_from_csv(csv_file_name=self.csv_name, index_col=None)
        assert_frame_equal(df, df2)

    @logit()
    def test_write_df_to_csv_empty(self):
        empty_df = PandasUtil.empty_df()
        expected_log_message = 'Empty dataframe will not be written.'
        with self.assertLogs(PandasUtil.__name__, level='DEBUG') as cm:
            self.pu.write_df_to_csv(df=empty_df, csv_file_name=self.csv_name, write_index=False)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_write_df_to_parquet(self):
        df = self.my_test_df()
        parquet_file = self.spreadsheet_name + '.parquet'
        # parquet creates an index named index. Set an index of that name.
        idx = pd.Index(df.index.array, dtype=df.index.dtype, name='index')
        exp_df = df.set_index(idx, inplace=False)

        self.pu.write_df_to_parquet(df=df, parquet_file_name=parquet_file, compression=None)
        df2 = self.pu.read_df_from_parquet(parquet_file_name=parquet_file)
        assert_frame_equal(exp_df, df2)
        fu = FileUtil()
        fu.delete_file(parquet_file)

    @logit()
    def test_read_df_from_csv(self):
        df = self.my_test_df()
        self.pu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=False)
        df2 = self.pu.read_df_from_csv(csv_file_name=self.csv_name)
        assert_frame_equal(df, df2)

    @logit()
    def test_read_df_from_csv_with_index(self):
        df = self.my_test_df()
        self.pu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=True)
        df2 = self.pu.read_df_from_csv(csv_file_name=self.csv_name, index_col=0)
        assert_frame_equal(df, df2)

    @logit()
    def test_read_df_from_excel(self):
        df = self.my_test_df()
        self.pu.write_df_to_excel(df=df, excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)
        df2 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)
        assert_frame_equal(df, df2, "Test 1 fail.")
        # Test 2
        df3 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet='noSuchWorksheet')
        assert_frame_equal(PandasUtil.empty_df(), df3, "Test 2 fail.")
        # Test 3. Test for missing spreadsheet
        expected_log_message = 'Cannot find Excel file'
        with self.assertLogs(level=logging.WARN) as cm:
            df4 = self.pu.read_df_from_excel(excelFileName='NoSuchSpreadsheet.xls', excelWorksheet='noSuchWorksheet')
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False), "Test 3 fail: couldn't find message.")
            self.assertTrue(self.pu.is_empty(df4), "Test 3 fail: df isn't empty.")

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

    @logit()
    def test_select_blanks(self):
        df = self.my_test_df()
        record = 3
        df.at[record, 'Sex'] = '' # Change Robin's sex to blank
        expected_weight = df.at[record, 'Weight']
        expected_age = df.at[record, 'Age']
        logger.debug(f'Modified df with blank: {df.head()}')
        actual = self.pu.select_blanks(df, 'Sex')
        logger.debug(f'got dataframe with blanks: {actual.head()}')
        self.assertEqual(expected_weight, actual['Weight'].iloc[0])
        self.assertEqual(expected_age, actual['Age'].iloc[0])

    @logit()
    def test_select_non_blanks(self):
        df = self.my_test_df()
        record = 3
        df.at[record, 'Sex'] = '' # Change Robin's sex to blank
        actual = self.pu.select_non_blanks(df, 'Sex')
        logger.debug(f'original df: {df.head()}')
        logger.debug(f'returned non-blank df: {actual.head()}')
        df_exp = df[df.Sex != '']
        logger.debug(f'expected df: {df_exp.head()}')
        assert_frame_equal(df_exp, actual)

    def my_f(self, x:int) -> int:
        return x * x

    def test_add_new_col_with_func(self):
        def my_func() -> list:
            df = self.pu.df
            col_of_interest = df['number']
            return [self.my_f(x) for x in col_of_interest]
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        new_col_name = 'squared'
        df = self.pu.add_new_col_with_func(df, new_col_name, my_func)
        expected = my_func()
        actual = df[new_col_name].tolist()
        self.assertListEqual(actual, expected)

    @logit()
    def test_add_new_col_from_array(self):
        # Test 1. No index needed.
        df = self.my_test_df() # len 5
        new_vals = [x for x in range(94,115,5)] # also len 5
        actual = self.pu.add_new_col_from_array(df, 'IQ', new_vals)
        self.assertListEqual(new_vals, actual.IQ.tolist())

    @logit()
    def test_drop_col(self):
        df = self.my_test_df()
        before_drop = self.pu.get_df_headers(df) # should be ['Name', 'Weight', 'Age']
        # pick one to drop.
        col_to_drop = before_drop[1]
        after_drop = before_drop
        self.pu.drop_col(df, columns=col_to_drop, is_in_place=True)
        after_drop.remove(col_to_drop)
        self.assertListEqual(self.pu.get_df_headers(df), after_drop)

    def test_drop_row_by_criterion(self):
        _AGE_COL_NAME = 'Age'
        # Test 1, is_in_place = False
        df = self.my_test_df()
        before_drop_ages = df[_AGE_COL_NAME]
        max_age, _ = self.cu.list_max_and_min(before_drop_ages)
        exp_ages = self.cu.remove_all_occurrences(before_drop_ages, max_age)
        act_df = self.pu.drop_row_by_criterion(df, _AGE_COL_NAME, max_age, is_in_place=False)
        self.assertListEqual(exp_ages, list(act_df[_AGE_COL_NAME]), "Test 1 fail")
        # Test 2, is_in_place = True
        df = self.my_test_df()
        before_drop_ages = df[_AGE_COL_NAME]
        _, min_age = self.cu.list_max_and_min(before_drop_ages)
        exp_ages = self.cu.remove_all_occurrences(before_drop_ages, min_age)
        self.pu.drop_row_by_criterion(df, _AGE_COL_NAME, min_age, is_in_place=True)
        self.assertListEqual(exp_ages, list(df[_AGE_COL_NAME]), "Test 2 fail")

    def test_drop_row_if_nan(self):
        # From https://www.geeksforgeeks.org/how-to-drop-rows-with-nan-values-in-pandas-dataframe/
        _INT1_COL_NAME = 'Integers_1'
        _INT2_COL_NAME = 'Integers_2'
        nums = {_INT1_COL_NAME: [10,  15, 30, 40,  55, nan, 75, nan, 90, 150, nan],
                _INT2_COL_NAME: [nan, 21, 22, 23, nan,  24, 25, nan, 26, nan, nan]}
        df = self.pu.convert_dict_to_dataframe(nums)
        # Test 1, just Integers_1
        int1 = nums[_INT1_COL_NAME]
        exp1 = [x for x in int1 if not isnan(x)]
        act1 = self.pu.drop_row_if_nan(df, [_INT1_COL_NAME], is_in_place=False)
        self.assertListEqual(exp1, list(act1[_INT1_COL_NAME]), "Test 1 fail")
        # Test 2: all of a row is nan.
        int2 = nums[_INT2_COL_NAME]
        exp2 = [x for x, y in zip(int1, int2) if not (isnan(x) & isnan(y))] # remove rows where both are nan.
        act2 = self.pu.drop_row_if_nan(df, column_names=None, is_in_place=False)
        for exp, act in zip(exp2, list(act2[_INT1_COL_NAME])):
            if isnan(exp):
                self.assertTrue(isnan(act), "Test 2 fail: Both elements should be NaN.")
            else:
                self.assertEqual(exp, act, f'Test 2 fail: expected {exp} not equal to actual {act}')



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

    def test_drop_col_keeping(self):
        df = self.my_test_df()
        df2 = self.my_test_df()
        orig_cols = self.pu.get_df_headers(df) # Should be ['Weight', 'Name', 'Sex', 'Age']
        keep_cols = orig_cols[0] # Should be 'Weight'
        drop_cols = orig_cols[1:] # Should be ['Name', 'Sex', 'Age']
        expected = self.pu.drop_col(df, drop_cols, is_in_place=False)
        actual = self.pu.drop_col_keeping(df2, keep_cols, is_in_place=False)
        assert_frame_equal(expected, actual)

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
    def test_replace_col_names_by_pattern(self):
        df = self.my_test_df()
        cols = self.pu.get_df_headers(df)
        prefix = 'col'
        df2 = self.pu.replace_col_names_by_pattern(df, prefix, is_in_place=False)
        exp = [f'{prefix}{i:02d}' for i in range(len(cols))]
        act = self.pu.get_df_headers(df2)
        self.assertListEqual(exp, act)

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
    def test_replace_col_using_mult_cols(self):
        cols = ['Sex', 'Weight']
        df = self.my_test_df()
        def boys_gain_girls_lose(cols: list):
            sex = cols[0]
            weight = cols[1]
            if sex == 'male':
                return 1.10 * weight
            else:
                return 0.90 * weight
        expected_weights = df[cols].apply(boys_gain_girls_lose, axis=1)
        logger.debug(f'expected weights: {expected_weights}')
        df2 = self.pu.replace_col_using_mult_cols(df, column_to_replace='Weight', cols=cols, func=boys_gain_girls_lose)
        logger.debug(f'new weights: {df2.head()}')
        self.assertListEqual(expected_weights.tolist(), df2['Weight'].tolist())


    @logit()
    def test_replace_col_with_scalar(self):
        df = self.my_test_df()
        # Test 1. First, replace all ages.
        target_age = 30
        self.pu.replace_col_with_scalar(df, 'Age', target_age)
        for age in df['Age']:
            self.assertEqual(target_age, age)
        # Test 2. Using a Series mask.
        mask_series = self.pu.mark_rows_by_criterion(df, 'Sex', 'female')
        logger.debug(f'marked rows are: {mask_series}')
        new_target_age = 21
        self.pu.replace_col_with_scalar(df, 'Age', new_target_age, mask_series)
        logger.debug(f'replaced ages df: {df.head()}. tested mask_series of type {type(mask_series)}')
        def expected_ages(female_age:int) -> list:
            ans = [target_age if x == 'male' else female_age for x in df['Sex']]
            logger.debug(f'returning expected ages vector of: {ans}')
            return ans
        self.assertListEqual(expected_ages(new_target_age), df['Age'].tolist())
        # Test 3. using a list mask.
        newer_target_age = 22
        mask_list = mask_series.tolist()
        self.pu.replace_col_with_scalar(df, 'Age', newer_target_age, mask_list)
        self.assertListEqual(expected_ages(newer_target_age), df['Age'].tolist())
        logger.debug(f'replaced ages df: {df.head()}. tested mask_list of type {type(mask_list)}')
        # Test 4. Using a wrong type of mask.
        with self.assertLogs(PandasUtil.__name__, level='DEBUG') as cm:
            expected_log_message = 'mask must be None, a series, or a list'
            mask_wrong = {}
            self.pu.replace_col_with_scalar(df, 'Age', newer_target_age, mask_wrong)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_dummy_var_df(self):
        df = self.my_test_df()
        col_to_dummy = 'Sex'
        dummy_df = self.pu.dummy_var_df(df, col_to_dummy)
        logger.debug(f'Here is the dummy series: {dummy_df.head()}')
        cols = self.pu.get_df_headers(dummy_df)
        new_col = cols[-1]
        expected = col_to_dummy + '_'
        self.assertTrue(next((True for col in cols if expected in col), False)) # must find a Sex_male or Sex_female column
        def expected_male() -> list:
            ans = [1 if x == 'male' else 0 for x in df['Sex']]
            return ans
        self.assertListEqual(expected_male(), dummy_df[new_col].tolist())

    @logit()
    def test_coerce_to_string(self):
        # Test 1. Single column (by name)
        df = self.my_test_df()
        age_series = df['Age']
        df2 = self.pu.coerce_to_string(df, 'Age')
        age_str_series = df2['Age']
        expected = list(df['Age'])
        actual = list(age_str_series)
        self.assertListEqual(expected, actual)
        # Test 2. Several columns (as a list)
        df3 = self.pu.coerce_to_string(df, ['Age', 'Name'])
        actual = list(df3['Name'])
        expected = list(df['Name'])
        self.assertListEqual(expected, actual)

    @logit()
    def test_coerce_to_numeric(self):
        # Test 1: coerce an integer. Using df in-place.
        df = self.my_test_df()
        age_series = df['Age'].astype(str).astype(int)
        self.pu.coerce_to_numeric(df, 'Age')
        age_num_series = df['Age']
        expected = age_series.tolist()
        actual = list(age_num_series)
        self.assertListEqual(expected, actual, "Fail test 1")
        # Test 2: coerce two columns. Using returned df.
        weight_series = df['Weight'].astype(str).astype(int)
        df3 = self.pu.coerce_to_numeric(df, ['Age', 'Weight'])
        weight_num_series = df3['Weight']
        self.assertListEqual(weight_series.tolist(), weight_num_series.tolist(), "Fail test 2")
        # Test 3: coerce strings and ints
        sizes = [4.0, 5.5, '5.5', 10.5, 8]
        col_name = 'Shoe_size'
        df = pd.DataFrame({col_name: sizes})
        exp = [float(x) for x in sizes]
        self.pu.coerce_to_numeric(df, col_name)
        act = list(df[col_name])
        self.assertListEqual(exp, act, 'Fail test 3')

    def is_adult(self, age:list):
        return age >= 21

    @logit()
    def test_mark_rows_by_func(self):
        df = self.my_test_df()
        mark = self.pu.mark_rows_by_func(df, 'Age', self.is_adult)
        logger.debug(f'marked rows are: {mark}')
        ans = [x >= 21 for x in df['Age']]
        self.assertListEqual(ans, mark.tolist())

    @logit()
    def test_mark_rows_by_criterion(self):
        df = self.my_test_df()
        mark = self.pu.mark_rows_by_criterion(df, 'Sex', 'female')
        logger.debug(f'marked rows are: {mark}')
        expected = [x == 'female' for x in df['Sex']]
        self.assertListEqual(expected, mark.tolist())

    @logit()
    def test_replace_vals_by_mask(self):
        df = self.my_test_df()
        mark = self.pu.mark_rows_by_criterion(df, 'Sex', 'male')
        logger.debug(f'marked rows are: {mark}')
        new_name = "Billy Bob"
        self.pu.replace_vals_by_mask(df, mark, col_to_change='Name', new_val=new_name)
        new_mark = self.pu.mark_rows_by_criterion(df, 'Name', new_name)
        # self.assertListEqual(mark, new_mark)
        # self.assertSequenceEqual(mark, new_mark)
        for male, bob in zip(mark, new_mark):
            self.assertFalse(male ^ bob)

    @logit()
    def test_mark_isnull(self):
        df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        expected_dict = self.list_of_dicts[-1]   # should be {name: '', number: -1}
        logger.debug(f'The expected entry is {expected_dict}')
        actual = self.pu.mark_isnull(df, 'name')
        logger.debug(f'mask of isnull: {actual}')
        self.assertTrue(actual.iloc[-1])
        for i in range(len(actual)-1):
            self.assertFalse(actual.iloc[i])

    @logit()
    def test_masked_df(self):
        # Test 1
        df = self.my_test_df()
        mask = [x >= 21 for x in df['Age']]
        expected = df[mask]
        actual = self.pu.masked_df(df=df, mask=mask, invert_mask=False)
        logger.debug(f'masked rows are: {actual.head()}')
        assert_frame_equal(expected, actual)
        # Test 2. Test inverted mask
        df_male = self.pu.select(df, 'Sex', 'male')
        mask_series = self.pu.mark_rows_by_criterion(df, 'Sex', 'female')
        actual = self.pu.masked_df(df=df, mask=mask_series, invert_mask=True)
        assert_frame_equal(df_male, actual)

    def test_slice_df(self):
        # Test 1, whole list
        lst = list(range(10,100,10)) # [10 .. 90]
        df = pd.Series(lst).to_frame()
        exp1 = list(range(10,100,10))
        df_exp1 = pd.Series(exp1).to_frame()
        df_act1 = self.pu.slice_df(df_exp1)
        assert_frame_equal(df_act1, df)
        # Test 2, start on second row and skip every other
        lst2 = [20, 40, 60, 80]
        df_exp2 = pd.Series(lst2).to_frame()
        df_act2 = self.pu.slice_df(df, start_index=1, step=2)
        self.pu.drop_index(df_exp2) # had indexes 0,1,2,3
        self.pu.drop_index(df_act2) # had indexes 1,3,5,7
        assert_frame_equal(df_exp2, df_act2)
        # Test 3, end early
        end_here = 5
        df3 = df[0:end_here]
        df_act3 = self.pu.slice_df(df, end_index=end_here)
        assert_frame_equal(df3, df_act3)
        # Test 4, all params
        start_here = 1
        my_step = 3
        df_exp4 = df.iloc[start_here:end_here:my_step]
        df_act4 = self.pu.slice_df(df=df, start_index=start_here, end_index=end_here, step=my_step)
        assert_frame_equal(df_exp4, df_act4)
        # Test 5, step = 1
        df_exp5 = df.copy()
        df_act5 = self.pu.slice_df(df=df, step=1)
        assert_frame_equal(df_exp5, df_act5)


    @logit()
    def test_is_empty(self):
        df = self.my_test_df()
        self.assertFalse(self.pu.is_empty(df))
        df2 = self.pu.empty_df()
        self.assertTrue(self.pu.is_empty(df2))

    def test_get_worksheets(self):
        # The majority of this is tested elsewhere, so just test for missing filename.
        expected_log_message = 'Cannot find Excel file'
        with self.assertLogs(PandasUtil.__name__, level='DEBUG') as cm:
            self.pu.get_worksheets(excelFileName='NoSuchFile.xlsx')
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_duplicate_rows(self):
        list_with_dups = self.list_of_dicts
        orig_df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_el = self.list_of_dicts[0]
        logger.debug(f'First element in list of dicts is: {first_el}')
        list_with_dups.append(first_el) # list has dup of first and last element
        expected = self.pu.convert_dict_to_dataframe([first_el])
        self.pu.reset_index(expected, True)
        logger.debug(f'Expected DF: {expected.head()}')
        df = self.pu.convert_dict_to_dataframe(list_with_dups)
        logger.debug(f'df with dupes is: {df.head()}')
        actual = self.pu.duplicate_rows(df)
        self.assertEqual(1, len(actual))

    @logit()
    def test_drop_duplicates(self):
        list_with_dups = self.list_of_dicts
        original_df = self.pu.convert_dict_to_dataframe(self.list_of_dicts)
        first_el = self.list_of_dicts[0]
        list_with_dups.append(first_el) # list has dup of first and last element
        extra_df = self.pu.convert_dict_to_dataframe(list_with_dups)
        actual = self.pu.drop_duplicates(extra_df)
        assert_frame_equal(original_df, actual)
        actual2 = self.pu.drop_duplicates(extra_df, fieldList=['name'])
        assert_frame_equal(original_df, actual2)

    @logit()
    def test_replace_col_names(self):
        df = self.my_test_df()
        orig_cols = self.pu.get_df_headers(df)
        new_cols = [f'{x}_new' for x in orig_cols]
        mapping_dict = dict(zip(orig_cols, new_cols))
        df2 = self.pu.replace_col_names(df=df, replace_dict=mapping_dict, is_in_place=False)
        actual_cols = self.pu.get_df_headers(df2)
        self.assertListEqual(new_cols, actual_cols)

    @logit()
    def test_aggregates(self):
        df = self.my_test_df()
        results_df = self.pu.aggregates(df, group_by=['Sex'], col='Weight')
        logger.debug(f'Results are: {results_df.head()}')
        df_female = self.pu.select(df, 'Sex', 'female')
        df_male = self.pu.select(df, 'Sex', 'male')
        expected_female_avg = df_female.agg("mean")['Weight']
        logger.debug(f'female mean weight is {expected_female_avg}')
        actual_female_row = results_df.loc[results_df['Sex'] == 'female']
        logger.debug(f'Results df for female row: {actual_female_row.head()}')
        actual_female_avg = actual_female_row['mean'].iloc[0]
        self.assertAlmostEqual(expected_female_avg, actual_female_avg)
        male_weights = df_male.Weight
        logger.debug(f'male weights are: {male_weights}')
        self.assertEqual(male_weights.max(), df_male.agg("max")["Weight"])
        self.assertEqual(male_weights.min(), df_male.agg("min")["Weight"])
        self.assertEqual(male_weights.sum(), df_male.agg("sum")["Weight"])

    @logit()
    def test_round(self):
        costs = [6.526666667, 5.332222222, 4.55, 6.3, 3, 5.330666667, 2.6128]
        for places in range(5):
            expected = [round(c, places) for c in costs]
            df = pd.DataFrame({'cost': costs})
            actual = self.pu.round(df, {'cost': places})
            logger.debug(f"testing rounding to {places} places. {expected} versus {actual['cost'].tolist()}")
            for exp, act in zip(expected, actual['cost']):
                appropriate_delta = pow(10.0, 0 - places) # Needed for the borderline case where 4.55 => 4.5 for round(4.5,2) and 4.6 for pd.round
                self.assertAlmostEqual(exp, act, msg=f'Numbers not within {appropriate_delta}', delta=appropriate_delta)

    @logit()
    def test_replace_vals(self):
        df = self.my_test_df()
        expected = ['M' if x == 'male' else 'F' for x in df['Sex']]
        self.pu.replace_vals(df=df, replace_me='male', new_val='M', is_in_place=True)
        self.pu.replace_vals(df=df, replace_me='female', new_val='F', is_in_place=True)
        self.assertListEqual(expected, df['Sex'].tolist())

    @logit()
    def test_stats(self):
        df = pd.DataFrame({'Weight': [45, 98, 113, 140, 165],
                           'Age': [8, 12, 14, 25, 55]})
        actual_slope, actual_intercept, actual_r = self.pu.stats(df, 'Age', 'Weight')

        def estimate_coeff(x:list, y:list):
            n = np.size(x)
            n2 = np.size(y)
            if n == n2:
                s_x = np.sum(x)
                s_y = np.sum(y)
                m_x = s_x * 1.0 / n
                m_y = s_y * 1.0 / n
                s_xy = np.sum(x*y)
                s_xx = np.sum(x*x)
                s_yy = np.sum(y*y)

                SS_xy = s_xy - n * m_x * m_y
                SS_xx = s_xx - n * m_x * m_x

                slope = SS_xy / SS_xx
                intercept = m_y - slope * m_x

                r = (n * s_xy - s_x * s_y) / (sqrt((1.0 * n * s_xx - s_x * s_x) * (1.0 * n * s_yy - s_y * s_y)))
                logger.debug(f'I think r-squared is: {r * r}')

                return slope, intercept, r
            else:
                logger.error(f'Sizes of x and y vectors do not match. x is of size {n}. y is of size {n2}. Returning None')
                return None
        x = df['Age']
        y = df['Weight']
        expected_slope, expected_intercept, expected_r = estimate_coeff(x, y)
        logger.debug(f'slope, intercept are: {expected_slope}, {expected_intercept}')
        self.assertAlmostEqual(expected_slope, actual_slope, delta=0.001)
        self.assertAlmostEqual(expected_intercept, actual_intercept, delta=0.001)
        self.assertAlmostEqual(expected_r, actual_r, delta=0.001)

    def test_head(self):
        df = self.my_test_df()
        head_len = 3
        actual = self.pu.head(df, head_len)
        logger.debug(f'Returned: \n{actual}')
        expected = df[:head_len]
        assert_frame_equal(expected, actual)

    def test_head_as_string(self):
        df = self.my_test_df()
        head_len = 2
        actual = self.pu.head_as_string(df, head_len)
        for i in range(head_len):
            self.assertTrue(df.iat[i, 1] in actual, f'Could not find {df.iat[i, 1]} in string')
        if head_len + 1 < len(df):
            self.assertFalse(df.iat[head_len+1, 1] in actual, f'Should not have found {df.iat[i, 1]} in string')

    def test_tail_as_string(self):
        df = self.my_test_df()
        tail_len = 2
        actual = self.pu.tail_as_string(df, tail_len)
        for i in range(len(df) - tail_len, len(df)):
            self.assertTrue(df.iat[i, 1] in actual, f'Could not find {df.iat[i, 1]} in string')

    def test_tail(self):
        # Test 1; tail is a subset of the df.
        df = self.my_test_df()
        row_count, col_count = self.pu.get_rowCount_colCount(df)
        tail_len = row_count - 1
        actual = self.pu.tail(df, tail_len)
        logger.debug(f'Returned: \n{actual}')
        expected = df[-tail_len:]
        assert_frame_equal(expected, actual)
        # test 2; tail attempts at a superset of the df (but gets just the df itself)
        tail_len = row_count + 10
        actual = self.pu.tail(df, tail_len)
        logger.debug(f'Returned: \n{actual}')
        expected = df
        assert_frame_equal(expected, actual)

    def test_get_basic_data_analysis(self):
        df = self.my_test_df()
        actual = self.pu.get_basic_data_analysis(df)
        lines = actual.split('\n')
        expected = "Data columns (total 4 columns)"
        self.assertTrue(any(line.startswith(expected) for line in lines))

    def test_get_get_quartiles(self):
        df = self.my_test_df()
        actual = self.pu.get_quartiles(df)

        actual_weight_mean = actual['Weight']['mean']
        expected_weight_mean = df['Weight'].mean()
        self.assertEqual(expected_weight_mean, actual_weight_mean)
        actual_age_mean = actual['Age']['mean']
        expected_age_mean = df['Age'].mean()
        self.assertEqual(expected_age_mean, actual_age_mean)
        actual_age_25th_percentile = df['Age'].quantile(q=.25)
        expected_age_25th_percentile = actual['Age']['25%']
        self.assertEqual(expected_age_25th_percentile, actual_age_25th_percentile)

    def test_largest_index(self):
        df = self.my_test_df()
        exp = df.index
        act_argmax, act_max = self.pu.largest_index(df)
        self.assertEqual(exp.sort_values(ascending=False)[0], act_max)
        exp_list = list(exp) # convert from Int64Index to a plain list.
        self.assertEqual(exp_list.index(act_max), act_argmax)

    def test_smallest_index(self):
        df = self.my_test_df()
        exp = df.index
        act_argmin, act_min = self.pu.smallest_index(df)
        self.assertEqual(exp.sort_values(ascending=True)[0], act_argmin)
        exp_list = list(exp) # convert from Int64Index to a plain list.
        self.assertEqual(exp_list.index(act_argmin), act_min)

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

class Test_PandasDateUtil(unittest.TestCase):
    isoFormat = "%Y-%m-%dT%H:%M:%S"
    _DATETIME = 'datetime' # name of a column
    _CLOSE = 'Close'

    def setUp(self):
        self.du = DateUtil()
        self.pdu = PandasDateUtil()
        self.cu = CollectionUtil()
        self.spreadsheet_name = 'small.xls'
        self.csv_name = 'small.csv'

    def my_price(self, day: int, hour: int) -> float:
        return day * 100 + hour

    def list_of_dicts_with_datetime_str(self, start_day:int=1, end_day:int=9, start_hr:int=9, end_hr:int=9, myFormat='%Y-%m-%d'):
        """
        Return a list of dictionaries with a datetime str and a float amount.
        :param start_day: day of month
        :param end_day: day of month
        :param start_hr: 24 hr of day
        :param end_hr: use the same as start_hr to have only daily
        :param myFormat: how to format the datetime str, e.g., '%Y-%m-%d'
        :return: list of dict
        """
        list_of_dicts = []
        if start_hr == end_hr: # Indicates only one hour per day; bump end_hr so that range (start_hr, end_hr) will run once.
            end_hr += 1
        for d in range (start_day, end_day):
            for h in range (start_hr, end_hr):
                dt = self.du.intsToDateTime(myYYYY=2020, myMM=9, myDD=d, myHH=h)
                list_of_dicts.append({self._DATETIME: self.du.asFormattedStr(dt, myFormat=myFormat), self._CLOSE: self.my_price(d, h)})
        return list_of_dicts

    def my_pdu_test_df(self, start_day:int = 1, end_day:int = 10, start_hr:int = 9, end_hr:int = 17, myFormat:str = "%Y-%m-%dT%H:%M:%S") -> pd.DataFrame:
        dicts = self.list_of_dicts_with_datetime_str(start_day, end_day, start_hr, end_hr, myFormat)
        df = self.pdu.convert_dict_to_dataframe(dicts)
        self.pdu.set_index(df=df, format=self.isoFormat, columns=self._DATETIME)
        return df

    def wfm_df(self):
        df = pd.DataFrame({
            "Date": ["2020-09-01", "2020-09-02", "2020-09-03", "2020-09-04", "2020-09-08", "2020-09-09", "2020-09-10", "2020-09-11", "2020-09-14", "2020-09-15", "2020-09-16", "2020-09-17", "2020-09-18", "2020-09-21", "2020-09-22", "2020-09-23", "2020-09-24", "2020-09-25", "2020-09-28", "2020-09-29", "2020-09-30"],
            "Open": [24.02, 24.01, 24.799999, 25, 24.280001, 24.01, 24.030001, 23.92, 24.379999, 24.959999, 24.870001, 25.23, 24.940001, 24.450001, 23.98, 23.780001, 22.959999, 23.120001, 23.99, 23.719999, 23.360001],
            "High": [24.34, 24.66, 25.360001, 25.190001, 24.549999, 24.07, 24.67, 24.32, 24.969999, 25.040001, 26, 25.43, 25.4, 24.52, 24.360001, 24.129999, 23.719999, 23.709999, 24.27, 23.719999, 23.870001],
            "Low": [23.74, 23.93, 24.299999, 24.25, 23.74, 23.700001, 23.860001, 23.709999, 24.23, 24.559999, 24.75, 24.9, 24.9, 23.719999, 23.530001, 22.83, 22.559999, 23.01, 23.76, 23.07, 23.25],
            self._CLOSE: [24.049999, 24.57, 24.52, 24.790001, 23.969999, 23.84, 23.950001, 24.27, 24.809999, 24.879999, 25.709999, 25.110001, 25.129999, 24.040001, 23.65, 22.83, 23.32, 23.639999, 23.82, 23.26, 23.51],
            "Adj Close": [24.049999, 24.57, 24.52, 24.790001, 23.969999, 23.84, 23.950001, 24.27, 24.809999, 24.879999, 25.709999, 25.110001, 25.129999, 24.040001, 23.65, 22.83, 23.32, 23.639999, 23.82, 23.26, 23.51],
            "Volume": [30541900, 40334000, 42349300, 48745000, 49082900, 49427900, 54211700, 34861100, 49787100, 41954800, 51768200, 51531300, 115153200, 56188400, 39839500, 45697700, 43329100, 30229900, 41103500, 38416300, 43058500],
        })
        self.pdu.set_index(df, columns="Date", is_in_place=True)
        return df


    def write_df_to_csv(self):
        df = self.my_pdu_test_df()
        self.pdu.write_df_to_csv(df=df, csv_file_name=self.csv_name, write_index=True)

    def test_to_Datetime_index(self):
        # Test 1. Send it datetimes.
        dates = []
        start_day = 1
        start_hr = 9
        end_day = 10
        end_hr = 17
        for d in range (start_day, end_day):
            for h in range (start_hr, end_hr):
                dates.append(self.du.intsToDateTime(myYYYY=2020, myMM=9, myDD=d, myHH=h))

        actual = self.pdu.to_Datetime_index(dates)
        min_datetime = self.du.intsToDateTime(myYYYY=2020, myMM=9, myDD=start_day, myHH=start_hr)
        self.assertEqual(min_datetime, actual[0], 'min does not match')
        max_datetime = self.du.intsToDateTime(myYYYY=2020, myMM=9, myDD=end_day-1, myHH=end_hr-1)
        self.assertEqual(max_datetime, actual[len(actual)-1], 'max does not match')

    @logit()
    def test_set_index(self):
        # Test 1. Datetimes as yyyy-mm-dd strings
        regFormat = "%Y-%m-%d"
        dicts = []
        start_day = 1
        start_hr = 9
        end_day = 10
        end_hr = 17
        first_dt = None
        # Using same start_hr and end_hr to only have one entry per day.
        dicts = self.list_of_dicts_with_datetime_str(start_day=start_day, end_day=end_day, start_hr=start_hr,
                                                     end_hr=start_hr, myFormat=regFormat)

        def create_and_test_datetime_indexed_df(dicts: list, my_format: str = '%Y-%m-%d'):
            df = self.pdu.convert_dict_to_dataframe(dicts)
            datetimes = [self.du.asDate(d[self._DATETIME], myFormat=my_format) for d in dicts]
            last_dt, first_dt = self.cu.list_max_and_min(datetimes)
            self.pdu.set_index(df, columns=self._DATETIME, is_in_place=True)
            _, act_val = self.pdu.smallest_index(df)
            self.assertEqual(first_dt, act_val, 'failed to find smallest index')
            _, act_val = self.pdu.largest_index(df)
            self.assertEqual(last_dt, act_val, 'failed to find largest index')

        logger.debug('about to test and create first df')
        create_and_test_datetime_indexed_df(dicts, regFormat)
        logger.debug('first df created and tested')
        # Test 2, Datetimes as yyyy-mm-dd hh:MM:ss (isoFormat) strings
        dicts = []
        # Using different start_hr and end_hr to ensure many entries per day.
        dicts = self.list_of_dicts_with_datetime_str(start_day=start_day, end_day=end_day, start_hr=start_hr,
                                                     end_hr=end_hr, myFormat=self.isoFormat)
        logger.debug('about to test and create second df')
        create_and_test_datetime_indexed_df(dicts, self.isoFormat)
        logger.debug('second df with ISO times created and tested')

    def test_read_df_from_csv(self):
        self.write_df_to_csv()
        df = self.pdu.read_df_from_csv(csv_file_name=self.csv_name, index_col=self._DATETIME)
        exp_df = self.my_pdu_test_df()
        assert_frame_equal(exp_df, df)

    def test_resample(self):
        # Test 1, resampling hourly data by days
        start_day = 1
        start_hr = 9
        end_day = 10
        end_hr = 17
        orig_df = self.my_pdu_test_df(start_day, end_day, start_hr, end_hr)
        act_df = self.pdu.resample(orig_df, column=self._CLOSE, rule='D')  # D means sample for calendar days.
        for i, d in enumerate(range(start_day, end_day)):
            prices = [self.my_price(d, h) for h in range(start_hr,end_hr)]
            exp = np.average(np.array(prices, dtype=int))
            self.assertEqual(exp, act_df[i])
        # Test 2, resampling hourly data using OHLC.
        ohlc_df = self.pdu.resample(orig_df, column='ohlc', rule='D')  # D means sample for calendar days.
        for i, d in enumerate(range(start_day, end_day)):
            prices = [self.my_price(d, h) for h in range(start_hr,end_hr)]
            np_prices = np.array(prices, dtype=int)
            l = np.amin(np_prices)
            h = np.amax(np_prices)
            row = ohlc_df.iloc[i]
            self.assertEqual(l, row[((self._CLOSE, 'low'))])
            self.assertEqual(h, row[((self._CLOSE, 'high'))])

    def test_sma(self):
        # Simple SMA
        df = self.wfm_df()
        window_size = 5
        ma_df = self.pdu.sma(df, window=window_size)
        closes = df[self._CLOSE]
        for i, row in enumerate(ma_df.iterrows()):
            if i < window_size - 1:
                self.assertTrue(isnan(row[1][self._CLOSE]))
            else:
                avg = closes[i-window_size+1:i+1].mean() # When i = 4, I'm looking for closes[0:5].
                self.assertAlmostEqual(avg, row[1][self._CLOSE], 6)

    def test_add_bollinger(self):
        # Bollinger bands
        df = self.wfm_df()
        window_size = 10
        bb_df = self.pdu.add_bollinger(df=df, window=window_size, column_name=self._CLOSE)

        bb_no_nan_df = self.pdu.drop_row_if_nan(bb_df, ['SMA'], is_in_place=False)

        self.assertEqual(len(df) - window_size + 1, len(bb_no_nan_df), "unexpected number of NaN rows dropped.")
        ma_df = self.pdu.sma(df, window=window_size)
        for i, row in enumerate(bb_df.iterrows()):
            if i < window_size - 1:
                self.assertTrue(isnan(row[1]['SMA']))
            else:
                self.assertAlmostEqual(bb_df.iloc[i]['SMA'], ma_df.iloc[i][self._CLOSE], 4)

    def test_add_sma(self):
        df = self.wfm_df()
        length = 10
        with_sma_col_df = self.pdu.add_sma(df, length=length, column_name=self._CLOSE, ma_column='SMA')
        sma_no_nan_df = self.pdu.drop_row_if_nan(with_sma_col_df, ['SMA'], is_in_place=False)
        ma_series = self.pdu.sma(df, window=length, col_name_to_average=self._CLOSE)
        for i, row in enumerate(with_sma_col_df.iterrows()):
            if i < length - 1:
                self.assertTrue(isnan(row[1]['SMA']))
            else:
                self.assertAlmostEqual(row[1]['SMA'], ma_series[i], 4)


    def test_filter_date_index_by_dates(self):
        df = self.wfm_df()
        days = list(df.index) # should have 21 days
        df_len = len(df)
        self.assertEqual(df_len, len(days))
        # Now pick the first three
        fourth_time = days[2+1] # t0, t1, t2, t3 and we're looking for strictly < t3 and should get t0, t1, t2
        first_three_df = self.pdu.filter_date_index_by_dates(df, start_date=self.du.intsToDateTime(myYYYY=1970, myMM=1, myDD=1), end_date=fourth_time)
        orig_first_three = df[0:3]
        assert_frame_equal(orig_first_three, first_three_df)
