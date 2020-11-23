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

"""
Interesting Python features:
* Does some dict comprehension in test_replace_col_names. 
* Uses mocking.
** Uses self.assertLogs to ensure that an error message is logged in the calling routine.
* Uses next to figure out if it found expected_log_message in cm.output:
** self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
* test_aggregates uses some numpy aggregates, such as sum, min, max, and mean.
* in test_select_blanks, found the first element in a Series with .iloc[0]
* in test_select_blanks, set an element within a dataframe using df.at
* in test_mark_isnull, used an iloc[-1] to get the last record
"""

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
                           'Sex' : ['male', 'female', 'male', 'female', 'male'],
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
        df2 = self.pu.empty_df()
        self.assertIsNone(self.pu.get_df_headers(df2))

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

        assert_frame_equal(df,df2)
        df3 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet='noSuchWorksheet')
        assert_frame_equal(PandasUtil.empty_df(), df3)
        # test for missing spreadsheet
        expected_log_message = 'Cannot find Excel file'
        with self.assertLogs(PandasUtil.__name__, level='DEBUG') as cm:
            df4 = self.pu.read_df_from_excel(excelFileName='NoSuchSpreadsheet.xls', excelWorksheet='noSuchWorksheet')
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
            self.assertTrue(self.pu.is_empty(df4))

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
        df = self.my_test_df()
        mask = [x >= 21 for x in df['Age']]
        expected = df[mask]
        actual = self.pu.masked_df(df=df, mask=mask, invert_mask=False)
        logger.debug(f'masked rows are: {actual.head()}')
        assert_frame_equal(expected, actual)

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