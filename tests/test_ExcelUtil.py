import logging
import pprint
import sys
from copy import copy
from typing import List
from unittest import TestCase, skip

import pandas as pd
from pandas.testing import assert_frame_equal
from dateutil.relativedelta import relativedelta

from ExcelUtil import ExcelUtil, ExcelCell, ExcelRewriteUtil, PdfToExcelUtilTabula, PdfToExcelUtilPdfPlumber, DfHelper
from ExecUtil import ExecUtil
from FileUtil import FileUtil
from DateUtil import DateUtil
from LogitUtil import logit
from PandasUtil import PandasUtil
from StringUtil import StringUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

Strings = List[str]

"""
Interesting Python features:
* Nested Test classes mirror the structure of the class under test.
** They do a super init.

"""

class TestExcelUtil(TestCase):
    spreadsheet_name = "first.xlsx"

    def __init__(self, *args, **kwargs):
        super(TestExcelUtil, self).__init__(*args, **kwargs)
        self._eu = ExcelUtil()
        self._pu = PandasUtil()
        self._fu = FileUtil()
        self._su = StringUtil()
        self.platform = ExecUtil.which_platform()
        logger.debug(f'You seem to be running {self.platform}.')
        self.path = r'c:\temp' if self.platform == 'Windows' else r'/tmp'
        self.parent_spreadsheet_name = self._fu.qualified_path(self.path, self.spreadsheet_name)
        self.worksheet_name = 'test'

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if ExecUtil.which_platform() == 'Windows' else '/tmp'
        fu.delete_file(fu.qualified_path(path, cls.spreadsheet_name))

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

    def test_load_spreadsheet(self):
        df_expected = self.my_test_df()
        self._pu.write_df_to_excel(df=df_expected, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        df_actual = self._eu.load_spreadsheet(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name)
        assert_frame_equal(df_expected, df_actual)

    def test_convert_from_A1(self):
        # Normal case
        test1 = "a15"
        exp1 = self._eu.row_col_to_ExcelCell(col=1, row=15)
        self.assertEqual(exp1, self._eu.convert_from_A1_to_cell(test1), "fail test 1")
        test2 = "AB93"
        exp_col = 28
        exp_row = 93
        act_row, act_col = self._eu.convert_from_A1(test2)
        self.assertEqual(exp_row, act_row, "fail test 2 (rows)")
        self.assertEqual(exp_col, act_col, "fail test 2 (cols)")

    def test_ExcelCell_to_A1(self):
        # Normal case
        tests = ['A2', 'C27', 'AA10', 'ZZ99']
        for test in tests:
            asExcel1 = self._eu.convert_from_A1_to_cell(test)
            self.assertEqual(test, self._eu.ExcelCell_to_A1(asExcel1))

    def test_row_col_to_A1(self):
        # Normal case
        tests = [ExcelCell(1, 2), ExcelCell(col=3, row=27), ExcelCell(col=27,row=10), ExcelCell(col=702, row=99)]
        expected = ['A2', 'C27', 'AA10', 'ZZ99']
        for exp, test in zip(expected, tests):
            self.assertEqual(exp, self._eu.row_col_to_A1(row=test.row, col=test.col), f'mismatch for {test}')

    def test_get_excel_rectangle_start_to(self):
        # Normal case: rows
        test_start_1 = "a5"
        end_row = 10
        test_end_1 = f'a{end_row}'
        exp1 = [self._eu.row_col_to_ExcelCell(row=row, col=1) for row in range(5, end_row + 1)]
        act1 = self._eu.get_excel_rectangle_start_to(test_start_1, test_end_1)
        self.assertEqual(exp1, act1, "fail test 1")
        # Normal case: cols
        start_col = "A"
        end_col = "I"
        start_col_as_int = ord(start_col) - ord("A") + 1
        end_col_as_int = ord(end_col) - ord("A") + 1
        start_row = 13
        test_start_2 = f'{start_col}{start_row}'
        test_end_2 = f'{end_col}{start_row}'
        exp2 = [self._eu.row_col_to_ExcelCell(col=col, row=start_row) for col in range(start_col_as_int, end_col_as_int + 1)]
        act2 = self._eu.get_excel_rectangle_start_to(test_start_2, test_end_2)
        self.assertListEqual(exp2, act2, "fail case 2")

    def test_get_values(self):
        df_first = self.my_test_df()
        self._pu.write_df_to_excel(df=df_first, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        # Normal case. A2:A6
        first = "A2"
        last = "A6"
        area1 = self._eu.get_excel_rectangle_start_to(first, last)
        exp_df = self.my_test_df()
        exp1 = list(exp_df['Weight'])
        df = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name)
        self._pu.drop_row_if_nan(df)
        act1 = self._eu.get_values(df=df, rectangle=area1)
        self.assertListEqual(exp1, act1, "fail normal case 1")
        # Normal case: a6:d6
        area2 = self._eu.get_excel_rectangle_start_to("A6", "D6")
        act2 = self._eu.get_values(df=df, rectangle=area2)
        exp2 = [71, 'Kia', 'male', 21]
        self.assertListEqual(exp2, act2, "fail normal case 2.")

    def test_get_excel_filename_and_worksheet(self):
        # Test 1, normal
        d1 = {'filename': 'abc.xls', 'worksheet': 'My worksheet'}
        act_fn_1, act_ws_1 = self._eu.get_excel_filename_and_worksheet(d1)
        self.assertEqual(d1['filename'], act_fn_1)
        self.assertEqual(d1['worksheet'], act_ws_1)
        # Test 2, missing filename
        d2 = {'worksheet': 'My worksheet'}
        act_fn_2, act_ws_2 = self._eu.get_excel_filename_and_worksheet(d2)
        self.assertIsNone(act_fn_2)
        self.assertEqual(d2['worksheet'], act_ws_2)

    def test_get_excel_filename_and_worksheet_and_range(self):
        # Test 1. normal
        d1 = {'filename': 'abc.xls', 'worksheet': 'My worksheet', 'range': 'A1:C13'}
        act_fn_1, act_ws_1, act_range_1 = self._eu.get_excel_filename_and_worksheet_and_range(d1)
        self.assertEqual(d1['filename'], act_fn_1, "Fail filename test 1")
        self.assertEqual(d1['worksheet'], act_ws_1, "Fail worksheet test 1")
        self.assertEqual(d1['range'], act_range_1, "Fail range test 1")
        # Test 2, missing worksheet
        d2 = {'filename': 'abc.xls'}
        _, act_ws_2, act_range_2 = self._eu.get_excel_filename_and_worksheet_and_range(d2)
        self.assertIsNone(act_ws_2, "Fail test 2 (worksheet)")
        self.assertIsNone(act_range_2, "Fail test 2 (range)")
        # Test 3, missing range
        d3 = {'filename': 'abc.xls', 'worksheet': 'My worksheet'}
        _, _, act_range_3 = self._eu.get_excel_filename_and_worksheet_and_range(d3)
        self.assertIsNone(act_range_3, "Fail test 3")


    def test_convert_range_to_cells(self):
        # Normal case. A2:A6
        first = "A2"
        last = "A6"
        test1 = first + ":" + last
        act_first, act_last = self._eu.convert_range_to_cells(test1)
        self.assertEqual(first, act_first, "fail case 1a")
        self.assertEqual(last, act_last, "fail case 1b")
        # Abnormal case. should not have only a single reference
        test2 = last
        act_first, act_last = self._eu.convert_range_to_cells(test2)
        self.assertIsNone(act_first, "fail case 2a")
        self.assertIsNone(act_last, "fail case 2b")

    def test_ExcelCell_to_row_col(self):
        col_as_int = 3
        row_as_int = 9
        test = ExcelCell(col=col_as_int, row=row_as_int)
        act_row, act_col = self._eu.ExcelCell_to_row_col(test)
        self.assertEqual(col_as_int, act_col)
        self.assertEqual(row_as_int, act_row)

    def test_get_excel_rectangle(self):
        excel_range = 'A3:A5'
        expected = [ExcelCell(1,3), ExcelCell(1,4), ExcelCell(1,5)]
        actual = self._eu.get_excel_rectangle(excel_range)
        self.assertListEqual(actual, expected)

    def test_get_repeat(self):
        # Test 1. Nothing in the dictionary means repeat = 1
        test_dict = {}
        act_repeat, act_subperiods = self._eu.get_repeat(test_dict)
        self.assertEqual(1, act_repeat)
        self.assertEqual('none', act_subperiods)
        # Test 2. Has a repeat but a missing subperiod.
        test_2_repeat = 7
        test_dict['repeat'] = test_2_repeat
        act_repeat, act_subperiods = self._eu.get_repeat(test_dict)
        self.assertEqual('equal', act_subperiods)
        self.assertEqual(test_2_repeat, act_repeat)
        # Test 3, testing 3 values
        repeats = [1, 6, 9]
        subperiods = ['equal', 'divided', 'divided']
        for test_rep, test_sub in zip(repeats, subperiods):
            d = {'repeat': test_rep, 'subperiod': test_sub}
            act_repeat, act_subperiods = self._eu.get_repeat(d)
            self.assertEqual(test_sub, act_subperiods, 'Fail test 3 with subperiods')
            self.assertEqual(test_rep, act_repeat, 'Fail test 3 with repeat factor')
        # Test 4, test for a missing subperiod

        test_dict = {'subperiod': 'no such subperiod', 'repeat': 1}
        act_repeat, act_subperiods = self._eu.get_repeat(test_dict)
        self.assertEqual('none', act_subperiods)

from ExcelUtil import ExcelCompareUtil

class TestExcelCompareUtil(TestExcelUtil):
    compare_spreadsheet_name = "third.xlsx"

    def __init__(self, *args, **kwargs):
        super(TestExcelCompareUtil, self).__init__(*args, **kwargs)
        self._ecu = ExcelCompareUtil()
        self.compare_spreadsheet_name = self._fu.qualified_path(self.path, self.compare_spreadsheet_name)

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if ExecUtil.which_platform() == 'Windows' else '/tmp'
        fu.delete_file(fu.qualified_path(path, cls.compare_spreadsheet_name))

    def test_epsilon(self):
        # Test 1, default epsilon.
        list1 = [1.0, 16.0, 256.0]
        excel_util_epsilon_default = 1.0e-8 # default epsilon is 1.0e-8
        small_eps = excel_util_epsilon_default / 2.0
        list2 = [x + small_eps for x in list1]
        self.assertTrue(self._ecu.identical(list1, list2), "fail test 1a (within epsilon)")
        list2[1] = list1[1] + excel_util_epsilon_default + small_eps / 2.0 # Should make it 16.0000000125
        self.assertFalse(self._ecu.identical(list1, list2), "fail test 1b (one number outside epsilon)")
        # Test 2, different epsilon
        test_ecu = ExcelCompareUtil(epsilon=1.0e-7)
        self.assertTrue(test_ecu.identical(list1, list2), "fail test 2 (all numbers within epsilon)")

    def test_epsilon_property(self):
        # Test 1, getter test using class setter
        eps = 1.0e-4
        test_ecu = ExcelCompareUtil(epsilon=eps)
        self.assertEqual(eps, test_ecu.epsilon, 'fail test 1')
        # Test 2, setter
        new_eps = 1.0e-2
        test_ecu.epsilon = new_eps
        self.assertEqual(new_eps, test_ecu.epsilon, 'fail test 2')

    def test_close_numbers(self):
        # Test 1, default epsilon.
        list1 = [1.0, 16.0, 256.0]
        excel_util_epsilon_default = 1.0e-8 # default epsilon is 1.0e-8
        small_eps = excel_util_epsilon_default / 2.0
        list2 = [x + small_eps for x in list1]
        self.assertTrue(self._ecu.close_numbers(list1, list2), "fail test 1a (within epsilon)")
        list2[1] = list1[1] + excel_util_epsilon_default + small_eps / 2.0 # Should make it 16.0000000125
        self.assertFalse(self._ecu.close_numbers(list1, list2), "fail test 1b (one number outside epsilon)")
        # Test 2, different epsilon
        self.assertTrue(self._ecu.close_numbers(list1, list2, epsilon=1.0e-7), "fail test 2 (all numbers within epsilon)")

    def test_identical(self):
        # Test 1, no scaling
        df = self.my_test_df()
        list1 = list(df.Weight)
        list2 = copy(list1)
        self.assertTrue(self._ecu.identical(list1, list2), "fail test 1")
        # Test 2, scale by float
        scale2 = 2.5
        f_scaled = [el / scale2 for el in list1]
        self.assertTrue(self._ecu.identical(list1, f_scaled, scale2), 'fail test 2')
        # Test 3, scale by int
        scale3 = 10
        f_scaled = [el / scale3 for el in list1]
        self.assertTrue(self._ecu.identical(list1, f_scaled, scale3), 'fail test 3')
        # Test 4, different list lengths
        list4 = copy(list1)
        list4.append(999)
        self.assertFalse(self._ecu.identical(list1, list4), 'fail test 4')
        # Although cm.output should have a warning, it seems to be blank.
        # self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
        # Test 5, epsilon
        list5 = [1.0, 16.0, 256.0]
        test_epsilon = 1.0e-7 # default epsilon is 1.0e-8
        small_eps = test_epsilon / 2.0
        list6 = [x + small_eps for x in list5]
        self.assertFalse(self._ecu.identical(list5, list6), 'fail test 5a')
        self.assertTrue(self._ecu.identical(list5, list6, epsilon=test_epsilon), 'fail test 5a')
        # Test 6, list type mismatch (list1 is str, list2 is int64)
        list6a = ['a', 'b']
        list6b = [1234, 5678]

        # with self.assertLogs(ExcelUtil.__name__, level='WARNING') as cm:
        self.assertFalse(self._ecu.identical(list6a, list6b))
            # self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))


    def test_identical_ints(self):
        # Test 1, identical
        list1 = [1, 1, 2, 3, 5, 8, 11]
        list2 = list1.copy()
        self.assertTrue(self._ecu.identical_ints(list1, list2), 'fail test 1')
        # Test 2, one different
        list2[1] = -1
        self.assertFalse(self._ecu.identical_ints(list1, list2), 'fail test 2')
        # Test 3, scaling
        factor = 3.0
        list3 = [x * factor for x in list1]
        self.assertTrue(self._ecu.identical_ints(list1, list3, scaling=1.0/factor))

    def test_identical_strings(self):
        # Test 1, identical
        list1 = ['alfa', 'bravo', 'charlie', 'delta']
        list2 = list1.copy()
        self.assertTrue(self._ecu.identical_strings(list1, list2), 'fail test 1')
        # Test 2, one different
        list2[0] = 'alpha'
        self.assertFalse(self._ecu.identical_strings(list1, list2), 'fail test 2')
        # Test 3, should succeed with 2 sig chars (because ALfa and ALpha would be the same)
        self.assertTrue(self._ecu.identical_strings(list1, list2, significant_characters=2), 'fail test 3')
        # Test 4, should fail with 3 sig chars (because ALFa and ALPha would be different)
        self.assertFalse(self._ecu.identical_strings(list1, list2, significant_characters=3), 'fail test 4')

    def test_compare_list_els_against_scalar(self):
        # Test 1, strings
        name = 'Judy'
        list1 = [name] * 3
        self.assertTrue(self._ecu.compare_list_els_against_scalar(list1, name), 'fail test 1')
        # Test 2, strings with a mismatch
        list1.append('Bill')
        self.assertFalse(self._ecu.compare_list_els_against_scalar(list1, name), 'fail test 2')
        # Test 3, floats (identical)
        num = 42.0
        list2 = [num] * 13
        self.assertTrue(self._ecu.compare_list_els_against_scalar(list2, num), 'fail test 3')
        # Test 4, floats (one within epsilon)
        eps = self._ecu.epsilon
        list2.append(num + eps / 2.0)
        self.assertTrue(self._ecu.compare_list_els_against_scalar(list2, num), 'fail test 4')
        # Test 5, floats (one within epsilon, one outside of epsilon)
        eps = self._ecu.epsilon
        list2.append(num + eps * 1.1)
        self.assertFalse(self._ecu.compare_list_els_against_scalar(list2, num), 'fail test 5')

    def test_compare_to_scalar(self):
        df_first = self.my_test_df()
        self._pu.drop_row_by_criterion(df_first, column_name='Sex', criterion='male', is_in_place=True)
        self._pu.write_df_to_excel(df=df_first, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        # Test 1, single cell to compare
        scalar = 88
        file_dict = {'filename': self.parent_spreadsheet_name, 'worksheet': self.worksheet_name, 'range': 'a2:a2'}
        self.assertTrue(self._ecu.compare_to_scalar(file_dict, scalar), 'fail test 1')
        # Test 2, multiple cells to compare
        scalar = 'female' # Dropped males, so only females left.
        file_dict['range'] = 'C2:C3'
        self.assertTrue(self._ecu.compare_to_scalar(file_dict, scalar), 'fail test 2')
        # Test 3, compare with the adjacent cell and should fail
        file_dict['range'] = 'b2:c2'
        self.assertFalse(self._ecu.compare_to_scalar(file_dict, scalar), 'fail test 3')

    def test_verify(self):
        # Test 1, skipping rows.
        df = self.my_test_df()
        self._pu.write_df_to_excel(df=df, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        # Now skip every other line
        df_even = df.iloc[::2]
        self._pu.write_df_to_excel(df=df_even, excelFileName=self.compare_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        first_dict = {'filename': self.parent_spreadsheet_name, 'worksheet': self.worksheet_name, 'range': 'a2:a6', 'step': 2}
        second_dict = {'filename': self.compare_spreadsheet_name, 'worksheet': self.worksheet_name, 'range': 'a2:a4'}
        self.assertTrue(self._ecu.verify(first_dict, second_dict))
        # Test 2, with epsilon (and identical)
        test_eps = 0.01
        second_dict['epsilon'] = test_eps
        self.assertTrue(self._ecu.verify(first_dict, second_dict))
        # Test 3, rewrite file with a larger number that is twice epsilon.
        ###
        ### Following test removed because the spreadsheet is buffered (and it won't read in).
        # df2 = self.my_test_df()
        # df2.iloc[2, df2.columns.get_loc('Weight')] += test_eps + test_eps # Make sure the iloc points to a weight that won't be skipped.
        # self._pu.write_df_to_excel(df=df2, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        # first_dict = {'filename': self.parent_spreadsheet_name, 'worksheet': self.worksheet_name, 'range': 'a2:a6', 'step': 2}
        # second_dict = {'filename': self.compare_spreadsheet_name, 'worksheet': self.worksheet_name, 'range': 'a2:a4'}
        # self.assertFalse(self._ecu.verify(first_dict, second_dict))

    def test_logging(self):
        # Test 1, normal.
        lines = ['line 1', 'line 2', 'line 3', 'line 4']
        for line in lines:
            self._ecu.add_log_line(line)
        self.assertListEqual(lines, self._ecu.compare_log)

    @skip("Only run from parent test")
    def test_get_values(self):
        pass

class TestExcelRewriteUtil(TestExcelUtil):
    fmt_spreadsheet_name = "second.xlsx"
    node_excel_name = "node.xlsx"

    def __init__(self, *args, **kwargs):
        super(TestExcelRewriteUtil, self).__init__(*args, **kwargs)
        self._rwu = ExcelRewriteUtil()
        self.formatting_spreadsheet_name = self._fu.qualified_path(self.path, self.fmt_spreadsheet_name)
        self.node_spreadsheet_name = self._fu.qualified_path(self.path, self.node_excel_name)

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if ExecUtil.which_platform() == 'Windows' else '/tmp'
        logger.warning(f'Did not delete {cls.spreadsheet_name}')
        fu.delete_file(fu.qualified_path(path, cls.fmt_spreadsheet_name))

    def format_test_df(self):
        df = pd.DataFrame({'Year': [2018, 2019, 2020, 2021],
                           'Era':  ['past', 'past', 'present', 'future'],
                           'Income': ['4,606.18', '4707.19', '4,808.20', '4909.21'],
                           'Margin' : ['18.18%', '19.19%', '20.20%', '21.21%'],
                           })
        return df

    @skip("Only run from parent test")
    def test_get_values(self):
        pass

    @logit()
    def test_write_df_to_excel(self):
        # Test 1, no formatting
        df = self.format_test_df()
        self._rwu.init_workbook_to_write() # Avoids AttributeError: 'NoneType' object has no attribute 'sheetnames'
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)
        df_act = self._pu.read_df_from_excel(excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, header=0, index_col=0)
        # The rwu.write_df_to_excel put out a blank row after the headers. Delete it.
        ok_mask = self._pu.mark_isnull(df_act, 'Year')
        df_act = self._pu.masked_df(df_act, ok_mask, invert_mask=True)

        ecu = ExcelCompareUtil()
        self.assertTrue(ecu.identical(df['Income'], df_act['Income']))
        self.assertTrue(ecu.identical(df['Year'], df_act['Year']))

        # Test 2, formatting.
        self._rwu.init_workbook_to_write() # Keeps from having a second copy of the table
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True, attempt_formatting=True)
        df_act = self._pu.read_df_from_excel(excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, header=0, index_col=0)
        # The rwu.write_df_to_excel put out a blank row after the headers. Delete it.
        ok_mask = self._pu.mark_isnull(df_act, 'Year')
        df_act = self._pu.masked_df(df_act, ok_mask, invert_mask=True)
        exp_inc = [self._su.as_float_or_int(x) for x in df['Income']]
        self.assertListEqual(exp_inc, list(df_act['Income']))

    def test_write_df_to_new_ws(self):
        # Test 1, no formatting
        df = self.format_test_df()
        ws1_name = "test1"
        ws = self._rwu.write_df_to_new_ws(df, excelWorksheet=ws1_name, attempt_formatting=False)
        act1 = self._rwu.get_cells(ws, 'D1:D4') # These are the margins
        exp1 = list(df['Margin'])
        self.assertListEqual(exp1, act1)

    def test_set_active(self):
        # Test 1, setting active an existing worksheet.
        df = self.format_test_df()
        ws1_name = "test1"
        self._rwu.write_df_to_new_ws(df, excelWorksheet=ws1_name, attempt_formatting=False)
        self.assertTrue(self._rwu.set_active(ws1_name))
        # Test 2. Does not exist.
        ws2_name = "NoSuchWorksheetXX"
        self.assertFalse(self._rwu.set_active(ws2_name))

    def test_get_cell(self):
        # Test 1
        df = self.format_test_df()
        ws1_name = "test1"
        ws = self._rwu.write_df_to_new_ws(df, excelWorksheet=ws1_name, attempt_formatting=False)
        act1 = self._rwu.get_cell(ws, 'A1')
        exp1 = df.iloc[0][0]
        self.assertEqual(exp1, act1)
        act2 = self._rwu.get_cell(ws, 'C2')
        exp2 = df.iloc[1][2]
        self.assertEqual(exp2, act2)

    def test_get_cells(self):
        # Test 1
        df = self.format_test_df()
        ws1_name = "test1"
        ws = self._rwu.write_df_to_new_ws(df, excelWorksheet=ws1_name, attempt_formatting=False)
        act1 = self._rwu.get_cells(ws, 'C1:C4') # These are the Incomes
        exp1 = list(df['Income'])
        self.assertListEqual(exp1, act1)

    def test_copy_spreadsheet_to_ws(self):
        # Test 1, normal
        df_expected = self.my_test_df()
        self._pu.write_df_to_excel(df=df_expected, excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)

        df = self.format_test_df()
        ws_copy_name = 'copy'
        ws = self._rwu.copy_spreadsheet_to_ws(sourceFileName=self.parent_spreadsheet_name,
                                              sourceWorksheet=self.worksheet_name, destWorksheet=ws_copy_name, header=0,
                                              write_header=True)
        self._rwu.save_workbook(filename=self.formatting_spreadsheet_name)
        actual_df = self._pu.read_df_from_excel(excelFileName=self.formatting_spreadsheet_name, excelWorksheet=ws_copy_name, header=0)
        assert_frame_equal(df_expected, actual_df)

    def test_init_template(self):
        # Test 1, normal
        df = self.format_test_df()
        ws_copy_name = 'test2'
        self._rwu.init_workbook_to_write()
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)
        self._rwu.init_template(template_excel_file_name=self.formatting_spreadsheet_name,
                                template_excel_worksheet=self.worksheet_name,
                                output_excel_file_name=self.parent_spreadsheet_name,
                                output_excel_worksheet=ws_copy_name)
        actual_df = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=ws_copy_name, index_col=0)
        self._pu.drop_row_if_nan(actual_df, is_in_place=True) # delete the blank row after the header
        actual_df.index = actual_df.index.astype(int) # index was float; make it an int
        self._pu.coerece_to_int(actual_df, 'Year') # was float; make it int64
        self._pu.coerece_to_int(df, 'Year')        # was int32; make it int64
        assert_frame_equal(df, actual_df)
        self.fail('in progress') # TODO

    def test_copy_ws_to_ws(self):
        # Test 1, normal
        df = self.format_test_df()
        ws_copy_name = 'formatted copy'
        self._rwu.init_workbook_to_write()
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True, attempt_formatting=False)
        source_ws = self._rwu.copy_spreadsheet_to_ws(sourceFileName=self.formatting_spreadsheet_name,
                                                     sourceWorksheet=self.worksheet_name)
        self._rwu.copy_ws_to_ws(ws_source=source_ws, ws_source_name=ws_copy_name )
        self._rwu.save_workbook(filename=self.parent_spreadsheet_name)

        actual_df = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=ws_copy_name, index_col=0)
        self._pu.replace_col_names(actual_df, {'Unnamed: 1': 'Year', 'Unnamed: 2':'Era', 'Unnamed: 3':'Income', 'Unnamed: 4': 'Margin'}, is_in_place=True)
        self._pu.drop_row_if_nan(actual_df, is_in_place=True) # delete the blank row after the header
        actual_df.index = actual_df.index.astype(int) # index was float; make it an int
        self._pu.coerece_to_int(actual_df, 'Year') # was float; make it int64
        self._pu.coerece_to_int(df, 'Year')        # was int32; make it int64
        assert_frame_equal(df, actual_df)

    def test_rewrite_worksheet(self):
        df = self.format_test_df()
        # Following writes to second.xlsx
        self._rwu.init_workbook_to_write() # Avoids AttributeError: 'NoneType' object has no attribute 'sheetnames'
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)

        # Test 1, one range
        # Make a copy of the file (from second to first.xlsx)
        self._fu.copy_file(source_file=self.formatting_spreadsheet_name, destination=self.parent_spreadsheet_name)
        col_d = ['4018', '4019', '4020', '4021']
        values = copy(col_d)
        ranges = ['d3:d6']
        self._rwu.rewrite_worksheet(excel_filename=self.parent_spreadsheet_name, excel_worksheet=self.worksheet_name, ranges=ranges, vals=values)
        self._rwu.save_workbook(filename=self.parent_spreadsheet_name)
        df = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, header=0,index_col=0)
        self._pu.drop_row_if_nan(df)
        exp = [float(x) for x in values]
        self.assertListEqual(exp, list(df['Income']), "Failing test 1")

        # Test 2, multiple ranges
        # Make a copy of the file
        self._fu.copy_file(source_file=self.formatting_spreadsheet_name, destination=self.parent_spreadsheet_name)
        values = copy(col_d)
        col_e = ['1.18', '1.19', '1.20', '1.21']
        values.extend(col_e)
        ranges = ['d3:d6', 'e3:e6']
        self._rwu.rewrite_worksheet(excel_filename=self.parent_spreadsheet_name, excel_worksheet=self.worksheet_name, ranges=ranges, vals=values)
        self._rwu.save_workbook(filename=self.parent_spreadsheet_name)
        df = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, header=0,index_col=0)
        self._pu.drop_row_if_nan(df)
        exp_d = [float(x) for x in col_d]
        self.assertListEqual(exp_d, list(df['Income']), "Failing test 2 (column d)")
        exp_e = [float(x) for x in col_e]
        self.assertListEqual(exp_e, list(df['Margin']), "Failing test 2 (column e)")

    def test_load_and_write(self):
        df = self.format_test_df()
        # Following writes to second.xlsx
        self._rwu.init_workbook_to_write() # This will avoid the AttributeError: 'NoneType' object has no attribute 'sheetnames'
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)

        # Test 1, no scaling
        in_file_dict = {
            'filename': self.formatting_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'B3:B6'
        }

        out_file_dict = {
            'filename': self.parent_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'A1:A4'
        }

        self._rwu.load_and_write(in_file_dict, out_file_dict)
        # the following reads in years 2018..2021 as the index.
        df2 = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, header=None, index_col=0)
        # df2.index.tolist() should be [2018, 2019, 2020, 2021, 2, 3]. The end (2 and 3) was left over from before.

        for act, exp in zip(df2.index.tolist(), df['Year']):
            self.assertEqual(act, exp)

        # Test 2, scaling
        out_file_dict['scaling'] = 100

        self._rwu.load_and_write(in_file_dict, out_file_dict)
        # the following reads in years 201800..202100 as the index.
        df3 = self._pu.read_df_from_excel(excelFileName=self.parent_spreadsheet_name, excelWorksheet=self.worksheet_name, header=None, index_col=0)
        # df3.index.tolist() should be [201800, 201900, 202000, 202100, 2, 3]. The end (2 and 3) was left over from before.
        scaled = [x * 100 for x in df['Year']]

        for act, exp in zip(df3.index.tolist(), scaled):
            self.assertEqual(act, exp)

    def test__equal_subperiod(self):
        # Test 1. Normal repeat
        test1 = [2, 4, 6, 8]
        repeat1 = 2
        exp1 = []
        for el in test1:
            for i in range(repeat1):
                exp1.append(el)
        self.assertListEqual(exp1, self._rwu._equal_subperiod(vals=test1, repeat=repeat1))

    def test__divided_subperiod(self):
        # Test 1. Normal div
        test1 = [2, 4, 6, 8]
        repeat1 = 2
        exp1 = []
        for el in test1:
            for i in range(repeat1):
                exp1.append(el / repeat1)
        self.assertListEqual(exp1, self._rwu._divided_subperiod(vals=test1, repeat=repeat1))

    def test__repeated(self):
        # Test 1. empty dictionary means repeat=1 and return the same list.
        d1 = {}
        test1 = [2, 4, 6, 8]
        act1 = self._rwu._repeated(d1, test1)
        self.assertListEqual(test1, act1)
        # Test 2. Has a repeat but a missing subperiod.
        test_2_repeat = 3
        d2 = {'repeat': test_2_repeat}
        test2 = test1
        exp2 = []
        for el in test2:
            for i in range(test_2_repeat):
                exp2.append(el)
        self.assertListEqual(exp2, self._rwu._repeated(d2, test2))
        # Test3. Has both a repeat and a 'divided' subperiod.
        test_3_repeat = 2
        d3 = {'repeat': test_3_repeat, 'subperiod': 'divided'}
        test3 = test1
        exp3 = []
        for el in test3:
            for i in range(test_3_repeat):
                exp3.append(el / test_3_repeat)
        self.assertListEqual(exp3, self._rwu._repeated(d3, test3))

    def test_read_template_and_nodes(self):
        # Test 1. Normal.
        df = self.format_test_df()
        # Following writes to second.xlsx
        self._rwu.init_workbook_to_write()
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)
        # Following writes to a node that will be read in by read_template_and_nodes.
        df_node = pd.DataFrame({
                           'name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
                           'score': [100, 200, 300, 400, 500]})
        self._pu.write_df_to_excel(df=df_node, excelFileName=self.node_spreadsheet_name, excelWorksheet=self.worksheet_name)

        in_file_dict = {
            'filename': self.formatting_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'B3:B6'
        }

        # output file is first.xlsx.
        out_file_dict = {
            'filename': self.parent_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'A1:A4'
        }
        # Node file is node.xlsx.
        node_file_dict = {
            'filename': self.node_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'b2:b6',
            'outputrange': 'a7:e7'
        }
        dict_list = [node_file_dict]
        self._rwu.read_template_and_nodes(template_dict=in_file_dict, output_dict=out_file_dict, dicts=dict_list)
        # Now verify that the node file(s) were written
        ecu = ExcelCompareUtil()
        for file_dict in dict_list:
            exp_values = df_node['score'].values.tolist()
            act_values = self._rwu.get_spreadsheet_values(file_dict)
            self.assertTrue(ecu.identical(exp_values, act_values))

    def test_stream_df_to_ws(self):
        # Test 1. Normal.
        df = self.format_test_df()
        # Following writes to second.xlsx
        self._rwu.init_workbook_to_write()
        self._rwu.write_df_to_excel(df, excelFileName=self.formatting_spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=True, write_header=True)
        # Following creates a df that will be streamed to the existing workbook.
        df = pd.DataFrame({
                           'name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
                           'score': [100, 200, 300, 400, 500]})
        self._rwu.stream_df_to_ws(df, self.worksheet_name)
        self._rwu.save_workbook(self.parent_spreadsheet_name) # Writes to first.xlsx
        # Now test to see if the df wrote out ok.
        exp_names = df['name'].values.tolist()
        # output file is first.xlsx.
        out_file_dict = {
            'filename': self.parent_spreadsheet_name,
            'worksheet': self.worksheet_name,
            'range': 'b9:b13'
        }
        ecu = ExcelCompareUtil()
        act_values = self._rwu.get_spreadsheet_values(out_file_dict)
        self.assertTrue(ecu.identical(exp_names, act_values), 'Fail test 1 (names)')
        exp_scores = df['score'].values.tolist()
        out_file_dict['range'] = 'c9:c13'
        act_values = self._rwu.get_spreadsheet_values(out_file_dict)
        self.assertTrue(ecu.identical(exp_scores, act_values), 'Fail test 1 (scores)')


"""
This class tests tabula-py.
"""
class TestPdfToExcelTabula(TestExcelUtil):
    csv_name = "convert.csv"
    def __init__(self, *args, **kwargs):
        super(TestPdfToExcelTabula, self).__init__(*args, **kwargs)
        self._pdf = PdfToExcelUtilTabula()
        self.converting_spreadsheet_name = self._fu.qualified_path(self.path, self.csv_name)

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if ExecUtil.which_platform() == 'Windows' else '/tmp'
        logger.warning(f'Did not delete {cls.spreadsheet_name}')
        fu.delete_file(fu.qualified_path(path, cls.csv_name))

    def test_read_pdf_tables(self):
        # This one uses tabula
        logger.debug('Using tabula-py.')
        # Test 1, local WF Annual report
        pdf_path = r"./2019-annual-report.pdf"
        df_list = self._pdf.read_pdf_table(pdf_path, pages=[1], read_many_tables_per_page=False)
        income_df = df_list[0]
        self.assertEqual(30, len(income_df))
        self._pu.replace_col_names_by_pattern(income_df, is_in_place=True)
        logger.debug(f'Income statement: \n {income_df.head(10)}')
        one_row = self._pu.select(income_df, column_name='col00', match_me='Wells Fargo net income $')
        logger.debug(f'selected row is :\n{one_row.head()}')
        self.assertEqual('19,549', one_row['col01'].any())

    def test_read_pdf_write_csv(self):
        # Test 1, Convert one page to CSV
        pdf_path = r"./2019-annual-report.pdf"
        result = self._pdf.read_pdf_write_csv(pdf_filename=pdf_path, csv_filename=self.converting_spreadsheet_name, pages = '2')
        self.assertTrue(result, "Failed test 1.")
        df = self._pu.read_df_from_csv(csv_file_name=self.converting_spreadsheet_name, header=1, enc='ISO-8859-1')
        self._pu.replace_col_names_by_pattern(df, is_in_place=True)
        logger.debug(f'Beginning of read-in CSV file: {df.head()}')
        one_row = self._pu.select(df, column_name='col00', match_me='Interest-earning deposits with banks')
        self.assertEqual('149,736', one_row['col03'].any())
        # Test 2, no such file (should throw a warning)
        pdf_path = r"./no_such_file.pdf"
        result = self._pdf.read_pdf_write_csv(pdf_filename=pdf_path, csv_filename=self.converting_spreadsheet_name, pages = '2')
        self.assertFalse(result, "Failed test 2.")

    @skip("Throwing unknown Tabula errors")
    def test_read_tiled_pdf_tables(self):
        # Test 1, read 4 tables and combine into one
        pdf_path = r"./2019-annual-report.pdf"
        df_list = self._pdf.read_tiled_pdf_tables(pdf_path, rows=2, cols=2, pages='all', tile_by_rows=True, read_many_tables_per_page=False)
        # self.fail('progress') #TODO

    @skip("Only run from parent test")
    def test_get_values(self):
        pass

"""
This class tests pdfplumber
"""
class TestPdfToExcelUtilPdfPlumber(TestExcelUtil):
    def __init__(self, *args, **kwargs):
        super(TestPdfToExcelUtilPdfPlumber, self).__init__(*args, **kwargs)
        self._pdf = PdfToExcelUtilPdfPlumber()

    @skip("Takes too long (about 90 seconds)")
    def test_read_pdf_table(self):
        logger.debug('Using pdfplumber.')
        logger.info('Disabling all logging but Info and higher for test_read_pdf_table!')
        logging.disable(logging.INFO)

        # Test 1, local WF Annual report
        pdf_path = r"./2019-annual-report.pdf"
        df_list = self._pdf.read_pdf_table(pdf_path, pages=[0])

        logging.disable(logging.DEBUG)
        logger.info('Enabling logging for test_read_pdf_table.')
        self.assertEqual(1, len(df_list))

    def test_summarize_pdf_tables(self):
        # Turn logging down (because PdfPlumber is very noisy with debug statements)
        logger.info('Disabling all logging but Info and higher for test_summarize_pdf_tables!')
        logging.disable(logging.INFO)

        pdf_path = r"./2019-annual-report.pdf"
        # Test 1. Single table.
        summary = self.summarize_single_table(pdf_path)
        self.assertTrue(any(line.find('***Table') >= 0 for line in summary))

        # Test 2. All tables
        summary = self.summarize_multiple_tables(pdf_path)
        self.assertTrue(any(line.find('***Table 2') >= 0 for line in summary))
        self.assertTrue(any(line.find('11,532,712') >= 0 for line in summary))
        self.assertTrue(any(line.find('19,549') >= 0 for line in summary))

        # Turn logging back on
        logging.disable(logging.DEBUG)
        logger.info('Enabling logging for test_summarize_pdf_tables.')
        logger.info('First lines of summary are:\n')
        for line in summary:
            logger.info(f'  {line}')

    def summarize_multiple_tables(self, pdf_path) -> Strings:
        # Test 2, read multiple tables
        summary = self._pdf.summarize_pdf_tables(pdf_path, pages='all')
        return summary

    def summarize_single_table(self, pdf_path: str) -> Strings:
        # Test 1, read a single table.
        # Separating into a function because it takes 2 min to run.
        # return ['***Table 0']  #remove this if you want it to run.
        summary = self._pdf.summarize_pdf_tables(pdf_path, pages=[0])
        return summary

    def test_read_tiled_pdf_tables(self):
        logger.debug('starting PdfPluber read_tiled_pdf_tables')
        # Turn logging down (because PdfPlumber is very noisy with debug statements)
        logger.info('Disabling all logging but Info and higher for test_read_tiled_pdf_tables!')
        logging.disable(logging.INFO)

        pdf_path = r"./2019-annual-report.pdf"
        df = self._pdf.read_tiled_pdf_tables(pdf_path, pages=[1,2,3,4,5,6], tables_to_tile=[3,4,5,6], rows=2, cols=2)
        print (df)
        # Turn logging back on
        logging.disable(logging.DEBUG)
        logger.info('Enabling logging for test_read_tiled_pdf_tables.')
        self.fail('in progress')

class TestDfHelper(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDfHelper, self).__init__(*args, **kwargs)
        self._df_helper = DfHelper()
        self._du = DateUtil()
        self._pu = PandasUtil()

    @logit()
    def test_set_column_names(self):
        # Test 1, column names supplied.
        cols = ['name', 'rank', 'serial_no']
        self._df_helper.set_column_names(col_name_list=cols)
        self.assertListEqual(cols, self._df_helper.column_names)
        # Test 2, both column names and col count supplied. Ignore col count and use column names.
        cols = ['Exactly', 'three', 'elements']
        self._df_helper.set_column_names(col_name_list=cols, col_count=8)
        self.assertListEqual(cols, self._df_helper.column_names)
        # Test 3, col count only.
        exp = ['col00', 'col01', 'col02', 'col03', 'col04', 'col05', 'col06', 'col07',]
        self._df_helper.set_column_names(col_count=8)
        self.assertListEqual(exp, self._df_helper.column_names)

    @logit()
    def test_add_row_and_clear(self):
        # Test 1, add a couple of columns
        df = pd.DataFrame({
            'name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
            'score': [100, 200, 300, 400, 500]})
        self._df_helper.set_column_names(col_name_list=['name', 'score'])
        names = df['name'].values.tolist()
        scores = df['score'].values.tolist()
        for name, score in zip(names, scores):
            self._df_helper.build_row(col_name='name', value=name)
            self._df_helper.build_row(col_name='score', value=score)
            self._df_helper.add_row_and_clear()
        act_df = self._df_helper.built_df()
        assert_frame_equal(df, act_df)

    @logit()
    def test_add_row_and_clear2(self):
        df = pd.DataFrame({
            'name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
            'score': [100, 200, 300, 400, 500]})
        names = df['name'].values.tolist()
        scores = df['score'].values.tolist()
        # Test 1, build the rows one at a time, using init_col_value and increment_col_value
        dates = [self._du.intsToDateTime(myYYYY=2021, myDD=1, myMM=mm) for mm in range(1, len(names) + 1)]
        self._pu.add_new_col_from_array(df, 'date', dates) # Adds a new col to df called 'dates' with '1/1/2021', '2/1/21' ... '5/1/21'
        aDate = self._du.intsToDateTime(myYYYY=2021, myDD=1, myMM=1)
        aScore = 100
        original = {'name': 'constant', 'score': aScore, 'date': aDate}
        self._df_helper.set_column_names(col_name_list=list(original.keys()))
        for col_name, value in original.items():
            self._df_helper.init_col_value(col_name=col_name, value=value)
        for name in names:
            self._df_helper.init_col_value('name', name) # to make it 'Sam', 'Adam', ... 'Kia'
            self._df_helper.add_row_and_clear()
            self._df_helper.increment_col_value('score', 100)
            self._df_helper.increment_col_value('date', 1, 'months')
        act_df = self._df_helper.built_df()
        assert_frame_equal(df, act_df)

    @logit()
    def test_init_col_value(self):
        # Test 1
        aDate = self._du.intsToDateTime(myYYYY=2021, myDD=1, myMM=1)
        exp_1 = {'monthNo': 1, 'name': 'constant', 'date': aDate}
        self._df_helper.set_column_names(col_name_list=list(exp_1.keys()))
        for col_name, value in exp_1.items():
            self._df_helper.init_col_value(col_name=col_name, value=value)
        self.assertEqual(exp_1, self._df_helper.staticColumn)

    @logit()
    def test_increment_col_value(self):
        aDate = self._du.intsToDateTime(myYYYY=2021, myDD=1, myMM=1)
        original = {'monthNo': 1, 'name': 'constant', 'date': aDate}
        self._df_helper.set_column_names(col_name_list=['monthNo', 'name', 'date'])
        for col_name, value in original.items():
            self._df_helper.init_col_value(col_name=col_name, value=value)
        # Test 1 - increment monthNo by 12
        monthNo_increment = 12
        self._df_helper.increment_col_value(col_name='monthNo', delta=monthNo_increment)
        act_dict = self._df_helper.staticColumn
        self.assertEqual(original['monthNo'] + monthNo_increment, act_dict['monthNo'], 'Fail test 1')
        # Test 2 - increment date by 1 month
        date_increment = 1 # and 'month'
        self._df_helper.increment_col_value(col_name='date', delta=date_increment, timePeriod='months')
        act_dict = self._df_helper.staticColumn
        exp_date = original['date'] + relativedelta(months=date_increment)
        self.assertEqual(exp_date, act_dict['date'], 'Fail test 2')
        # Test 3 - the name didn't get incremented and should be the same.
        act_dict = self._df_helper.staticColumn
        self.assertEqual(original['name'], act_dict['name'])
