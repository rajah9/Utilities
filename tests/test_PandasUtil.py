import pandas as pd
import numpy as np
import unittest
import logging
from math import sqrt
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
* test_replace_col_using_mult_cols uses an inner method to distinguish calculation of the expected value
* test_select_non_blanks deletes rows based on criteria.
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
    def test_join_dfs_on_index(self):
        df1 = self.my_test_df()

        def my_test_df2():
            df = pd.DataFrame({'Shoe_size': [4.0, 5.5, 5.5, 10.5, 8.0]})
            # Create and set the index
            index_ = [0, 1, 2, 3, 4]
            df.index = index_
            return df
        df2 = my_test_df2()
        actual = self.pu.join_dfs_on_index(df1, df2)
        for index, row in actual.iterrows():
            self.assertEqual(row["Shoe_size"], df2.iloc[index]["Shoe_size"])
            self.assertEqual(row["Name"], df1.iloc[index]["Name"])

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
        # Test 2
        df3 = self.pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet='noSuchWorksheet')
        assert_frame_equal(PandasUtil.empty_df(), df3)
        # Test 3. Test for missing spreadsheet
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