import logging
from unittest import mock, TestCase, main
import sys, pprint
import pandas as pd
from pandas.util.testing import assert_frame_equal
from ExcelUtil import ExcelUtil, ExcelCell
from ExecUtil import ExecUtil
from PandasUtil import PandasUtil
from LogitUtil import logit
from FileUtil import FileUtil
import sys, pprint
from copy import copy

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* Nested Test classes mirror the structure of the class under test.
** They do a super init.

"""

class TestExcelUtil(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestExcelUtil, self).__init__(*args, **kwargs)
        self._eu = ExcelUtil()
        self._pu = PandasUtil()
        self._fu = FileUtil()
        self.platform = ExecUtil.which_platform()
        logger.debug(f'You seem to be running {self.platform}.')
        self.path = r'c:\temp' if self.platform == 'Windows' else r'/tmp'
        self.spreadsheet_name = self._fu.qualified_path(self.path, 'first.xlsx')
        self.worksheet_name = 'test'

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
        self._pu.write_df_to_excel(df=df_expected, excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name, write_index=False)
        df_actual = self._eu.load_spreadsheet(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)
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
        # Normal case. A2:A6
        first = "A2"
        last = "A6"
        area1 = self._eu.get_excel_rectangle_start_to(first, last)
        df = self._pu.read_df_from_excel(excelFileName=self.spreadsheet_name, excelWorksheet=self.worksheet_name)
        act1 = self._eu.get_values(df=df, rectangle=area1)
        df = self.my_test_df()
        exp1 = list(df.Weight)
        self.assertListEqual(exp1, act1, "fail normal case 1")
        # Normal case: a6:d6
        area2 = self._eu.get_excel_rectangle_start_to("A6", "D6")
        act2 = self._eu.get_values(df=df, rectangle=area2)
        exp2 = [71, 'Kia', 'male', 21]
        self.assertEqual(exp2, act2, "fail normal case 2.")

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

from ExcelUtil import ExcelCompareUtil

class TestExcelCompareUtil(TestExcelUtil):
    def __init__(self, *args, **kwargs):
        super(TestExcelCompareUtil, self).__init__(*args, **kwargs)
        self._ecu = ExcelCompareUtil()
        pprint.pprint(sys.path)

    def test_identical(self):
        # Test 1, no scaling
        df = self.my_test_df()
        list1 = list(df.Weight)
        list2 = copy(list1)
        self.assertTrue(self._ecu.identical(list1, list2), "fail test 1")
        self.fail("in progress")

from ExcelUtil import PdfToExcelUtil

class TestPdfToExcel(TestExcelUtil):
    def __init__(self, *args, **kwargs):
        super(TestPdfToExcel, self).__init__(*args, **kwargs)
        self._pdf = PdfToExcelUtil()
        pprint.pprint(sys.path)

    def test_read_pdf_tables(self):
        # Test 1, local WF Annual report
        pdf_path = r"./2019-annual-report.pdf"
        df_list = self._pdf.read_pdf_tables(pdf_path, pages=[1, 2], read_many_tables_per_page=True)
        self.assertEqual(3, len(df_list)) # I only see 2 tables, but WF formatted as 3!
        income_df = df_list[1]
        balance_df = df_list[2]
        logger.debug(f'Balance statement: \n {balance_df.head(10)}')
        self.assertEqual(14, len(income_df))
        self._pu.replace_col_names_by_pattern(income_df, is_in_place=True)
        logger.debug(f'Income statement: \n {income_df.head(10)}')
        one_row = self._pu.select(income_df, column_name='col00', match_me='Wells Fargo net income')
        logger.debug(f'selected row is :\n{one_row.head()}')
        self.assertEqual('19,549', one_row['col02'].any())

    def test_read_tiled_pdf_tables(self):
        # Test 1, read 4 tables and combine into one
        pdf_path = r"./2019-annual-report.pdf"
        df_list = self._pdf.read_tiled_pdf_tables(pdf_path, rows=2, cols=2, pages='3-6', tile_by_rows=True, read_many_tables_per_page=False)
        self.fail('in progress') #TODO
