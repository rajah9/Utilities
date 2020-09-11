"""
These classes opens and compares two Excel worksheets. (May be from the same file.).
It also reads Excel spreadsheets, reads rows or columns, and writes them to another spreadsheet.

-ExcelUtil does helpful conversions, such as convert_from_a1 which converts C4 to the row, col pair of 4,3.
 It is the parent of these child classes:
 - ExcelCompareUtil - Compares two files
 - ExcelRewriteUtil - Rewrites part of an Excel spreadsheet

To import this library, here's a handy import statement:
from ExcelUtil import ExcelCompareUtil

# To create the class, here's a handy instantiation statement:
eu = ExcelCompareUtil()
"""

"""
Interesting Python features:
* Uses a namedtuple for ExcelCell.
* Main class inherits from Util and has two subclasses.
* Several functions return a Tuple of two ints.
* Uses functools.lru_cache to cache up to 4 spreadsheets. Very helpful to keep the same thing from being read twice.
* In replace_col_with_scalar, uses two different methods to replace a masked array with a scalar.
* In replace_col_using_func, I pass in a function to be executed by it.
* In stats, provided an independent way to find slope, intercept, and r.
* In the init, set the display options to show head() better.
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from Util import Util
from PandasUtil import PandasUtil
from StringUtil import StringUtil
from LogitUtil import logit
from CollectionUtil import CollectionUtil
from math import fabs
import functools
from copy import copy
from typing import List, Union, Tuple
import tabula
import pdfplumber

Strings = List[str]
Ints = List[int]
Floats = List[float]
Dataframes = List[pd.DataFrame]

_EPSILON = 1.0e-8
ExcelCell = namedtuple('ExcelCell', 'col row')

class ExcelUtil(Util):
    def __init__(self):
        super().__init__()
        self.logger.info('starting ExcelUtil')
        self._pu = PandasUtil()
        self._wb = None

    def row_col_to_ExcelCell(self, row:int, col:int) -> ExcelCell:
        """
        Convert the given row and column to an ExcelCell.
        :param row:
        :param col:
        :return:
        """
        return ExcelCell(col=col, row=row)

    def ExcelCell_to_row_col(self, ec: ExcelCell) -> Tuple[int, int]:
        """
        Convert an ExcelCell to a corresponding ExcelCell.
        :param ec:
        :return:
        """
        return ec.row, ec.col

    def convert_from_A1(self, convert_me: str) -> Tuple[int, int]:
        row_col = StringUtil(convert_me)
        col = row_col.excel_col_to_int()
        row = row_col.digits_only_as_int()
        return row, col

    def convert_from_A1_to_cell(self, convert_me: str) -> ExcelCell:
        """
        Convert a string like A2 to an ExcelCell
        :param range: string like A2
        :return: ExcelCell like Excel(1,2)
        """
        row, col = self.convert_from_A1(convert_me)
        return self.row_col_to_ExcelCell(row=row, col=col)

    def convert_range_to_cells(self, range: str) -> Tuple[str, str]:
        """
        Convert a range like "a2:a9" to a pair of cells
        :param range: str like "a2:a9"
        :return: Tuple of strings on either side of the colon, like ("a2", "a9")
        """
        str_range = StringUtil(range)
        cell_range = str_range.split_on_delimiter(delim=':')
        if len(cell_range) == 2:
            return cell_range[0], cell_range[1]
        self.logger.warn(f'Should have exactly one cell before a colon and one after, but is: {range}. Returning (None, None)')
        return None, None

    def get_excel_rectangle_start_to(self, start: str, to: str) -> list:
        """
        Return a rectangle of ExcelCells.
        :param start: str containing a cell, like "A2"
        :param to: str containing a cell, like "A9"
        :return:
        """
        start_row, start_col = self.convert_from_A1(start)
        end_row, end_col = self.convert_from_A1(to)
        ans = [self.row_col_to_ExcelCell(col=col, row=row) for row in range(start_row, end_row + 1) for col in range(start_col, end_col + 1)]
        return ans

    def get_excel_rectangle(self, excel_range: str) -> list:
        """
        Return a list of ExcelCells
        :param excel_range: str like "a2:a9"
        :return: list of ExcelCells
        """
        start, to = self.convert_range_to_cells(excel_range)
        return self.get_excel_rectangle_start_to(start, to)

    def get_values(self, df:pd.DataFrame, rectangle: list) -> list:
        ans = []
        for cell in rectangle:
            val = df.iloc[cell.row - 2, cell.col - 1] # row is -2 because of -1 for header for 0-offset. col is -1 b/c of 0-offset
            ans.append(val)
        return ans

    @functools.lru_cache(maxsize=4)
    def load_spreadsheet(self, excelFileName: str = None, excelWorksheet: str = None) -> pd.DataFrame:
        df = self._pu.read_df_from_excel(excelFileName = excelFileName, excelWorksheet=excelWorksheet)
        return df

    def get_spreadsheet_values(self, excel_file_dict: dict) -> list:
        """
        Return the values specified by the efd.filename, efd.worksheet, and efd.range
        :param excel_file_dict:
        :return:
        """
        df = self.load_spreadsheet(excelFileName=excel_file_dict['filename'], excelWorksheet=excel_file_dict['worksheet'])
        area = self.get_excel_rectangle(excel_file_dict['range'])
        vals = self.get_values(df, area)
        return vals

    def get_scaling(self, excel_file_dict: dict) -> float:
        """
        Read the excel_file_dict for the 'scaling' field. If there's none, return 1.0
        :param excel_file_dict:
        :return:
        """
        try:
            scaling = excel_file_dict['scaling']
        except KeyError:
            self.logger.debug("no scaling found; using 1.0")
            scaling = 1.0
        return scaling

"""
To import this library, here's a handy import statement:
from ExcelUtil import ExcelCompareUtil

# Handy instantiation statement:
eu = ExcelCompareUtil()

"""
class ExcelCompareUtil(ExcelUtil):
    def close_numbers(self, list1: Floats, list2: Floats, scaling: float = 1.0) -> bool:
        ans = True
        for el1, el2 in zip(list1, list2):
            if fabs(el1 - scaling * el2) > _EPSILON:
                ans = False
                self.logger.warning(f'mismatch: {el1} not equal to {el2} * scale {scaling}; difference of {fabs(el1 - el2 * scaling)}')
        return ans

    def identical_ints(self, list1: Ints, list2: Ints, scaling: float = 1.0):
        """
        determine if the list of integers is identical.
        :param list1:
        :param list2:
        :param scaling:
        :return:
        """
        ans = True
        for el1, el2 in zip(list1, list2):
            if el1 != scaling * el2:
                ans = False
                self.logger.warning(f'mismatch: {el1} not equal to {el2} * scale {scaling}; difference of {abs(el1 - el2 * scaling)}')
        return ans

    def identical_strings(self, list1: Strings, list2: Strings) -> bool:
        ans = True
        for el1, el2 in zip(list1, list2):
            if el1 != el2:
                ans = False
                self.logger.warning(f'mismatch: {el1} not equal to {el2}.')
        return ans

    def identical(self, list1: Strings, list2: Strings, scaling: Union[float, int] = 1) -> bool:
        if len(list1) != len(list2):
            self.logger.warning(f'Lists are different sizes. {len(list1)} not equal to {len(list2)})')
            return False

        list1_el = list1[0]
        list2_el = list2[0]
        if isinstance(list1_el, str):
            return self.identical_strings(list1, list2)
        elif isinstance(list2_el, (int, np.int64)):
            if isinstance(list1_el, str):
                self.logger.warning(f'first element of list1 is {list1_el}, but first element of list2 is {list2_el}. They cannot be compared. Returning False. ')
                return False
            return self.identical_ints(list1, list2, scaling)
        elif isinstance(list2_el, float):
            if isinstance(list1_el, str):
                self.logger.warning(f'first element of list1 is {list1_el}, but first element of list2 is {list2_el}. They cannot be compared. Returning False. ')
                return False
            return self.close_numbers(list1, list2, scaling)
        else:
            self.logger.warning(f'list2 should be str, int, or float, but was {type(list2_el)}. Returning False')
            return False

    def verify(self, file1: dict, file2: dict) -> bool:
        """
        Verify that the values in rectangles of file1 and file2 are identical.
        :param file1: dict with keys 'filename', 'worksheet', and 'range'
        :param file2:
        :return:
        """
        vals1 = self.get_spreadsheet_values(excelDict=file1)
        self.logger.debug(f'1: \n{vals1}')
        vals2 = self.get_spreadsheet_values(excelDict=file2)

        scaling = self.get_scaling(file2)

        self.logger.debug(f'2:\n{vals2}')
        ans = self.identical(vals1, vals2, scaling)
        report = 'identical' if ans else 'DIFFERENT'
        self.logger.info(f'lists are {report}')
        return ans


"""
Following routines are for reading and writing Excel files.
They use openpyxl.
See generates_spreadsheets.py as an example.
"""
from openpyxl import load_workbook, Workbook

class ExcelRewriteUtil(ExcelUtil):
    def load_and_write(self, file1: dict, file2: dict) -> bool:
        vals1 = self.get_spreadsheet_values(excelDict=file1)
        self.logger.debug(f'1: \n{vals1}')
        scaling = self.get_scaling(file2)
        scaled_vals = [v * scaling for v in vals1]
        self.rewrite_worksheet(file2, scaled_vals)
        return True

    @functools.lru_cache(maxsize=2)
    def init_workbook(self, filename: str) -> Workbook:
        self._wb = load_workbook(filename=filename)
        return self._wb

    @logit(showArgs=True, showRetVal=False)
    def save_workbook(self, filename: str):
        self._wb.save(filename=filename)

    @logit()
    def rewrite_worksheet(self, file: dict, vals: list):
        """
        Read in the given filename and worksheet (from the file dictionary).
        Write the values to the given range, preserving formatting.

        :param file: dictionary containing 'filename', 'worksheet', and 'range' keys.
        :param vals: list of values to be written. len(vals) should be the same as the range.
        :return:
        """
        wb = self.init_workbook(filename=file['filename'])
        ws = wb[file['worksheet']]
        # We have either a single range or many ranges to copy to.
        try:
            rectangles = file['ranges']
        except KeyError:
            range_rectangle = file['range']
            rectangles = [range_rectangle]
        for rectangle in rectangles:
            local_vals = copy(vals)
            excel_rect = self.get_excel_rectangle(rectangle) # Will return a list of ExcelCells
            if len(excel_rect) != len(vals):
                self.logger.warning(f'mismatch between source length (={len(vals)}) and target length (-{len(excel_rect)}) defined by cell range {rectangle}')
            start_cell, end_cell = excel_rect[0], excel_rect[-1]
            self.logger.debug(f'starting cell: {start_cell} / ending cell: {end_cell}')

            for row in range(start_cell.row, end_cell.row + 1):
                for col in range (start_cell.col, end_cell.col + 1):
                    val = local_vals.pop(0)
                    self.logger.debug(f'cell at row {row} / col {col} is {ws.cell(row=row, column=col, value=val).value}')
                return

class PdfToExcelUtil(ExcelUtil):
    def __init__(self):
        super().__init__()
        self.logger.info('starting PdfToExcelUtil')
        self._cu = CollectionUtil()
        self._su = StringUtil()

    @logit()
    def read_pdf_tables(self, pdf_filename: str, pages: Union[Ints, str] = 'all', make_NaN_blank: bool = True,
                        read_many_tables_per_page=False) -> Dataframes:
        """
        Using tabula-py, read the designated pages from the PDF.
        :param pdf_filename: full path and filename to PDF file. May be a URL.
        :param pages: page to read in (defaults to 'all')
        :param read_many_tables_per_page: False to read one table per page.
        :return: a list of dataframes, where each el is a table
        """
        self.logger.debug(f'About to read from {pdf_filename}')
#        dfs = tabula.read_pdf(pdf_filename, pages=pages, multiple_tables=read_many_tables_per_page)
        # tabula_build_opts = tabula.io.build_options(pages=pages, stream=True)
        dfs = tabula.read_pdf(pdf_filename, guess=False, pages=pages, stream=True, multiple_tables=read_many_tables_per_page, pandas_options={'header': None}) # problems with multiple_tables=read_many_tables_per_page)
        self.logger.debug(f'Read in {len(dfs)} tables.')
        if not make_NaN_blank:
            return dfs
        blanked_dfs = [df.replace(np.nan, '', regex=True) for df in dfs]
        return blanked_dfs

    def read_pdf_table(self, pdf_filename: str, pages: Ints) -> Dataframes:
        """
        Using pdfplumber, read the designated pages from the PDF.
        :param pdf_filename: full path and filename to PDF file. May be a URL.
        :param pages: page to read in (defaults to 'all')
        :param read_many_tables_per_page: False to read one table per page.
        :return: a list of dataframes, where each el is a table
        """
        self.logger.debug(f'About to read from {pdf_filename}')
        pdf = pdfplumber.open(pdf_filename)
        ans = []
        for p in pages:
            page = pdf.pages[p]
            table = page.extract_table()
            self.logger.debug(f'First 5 rows:\n{table[:5]}')
            df = pd.DataFrame(table[1:], columns=table[0])
            pass # TODO: Cleanup df
            ans.append(df)
        return ans



    def read_tiled_pdf_tables(self, pdf_filename: str, rows: int, cols: int, pages: Union[Ints, str] = 'all',
                              tables_to_tile: Ints = None, tile_by_rows: bool = True, read_many_tables_per_page=False,
                              make_NaN_blank: bool = True) -> Dataframes:
        """
        This reads the tables on several pages into a single table.
        :param pdf_filename:
        :param rows: How many rows across
        :param cols: How many columns down
        :param pages: Default: 'all'. Could be a str like "1,2,3" or a list like [1,2,3]. 1-offset.
        :param tile_by_rows: True if sequential pages comprise rows, False if they comprise columns.
        :param read_many_tables_per_page: False to read one table per page.
        :param make_NaN_blank: True to make NaN values blank.
        :return:
        """
        dfs = self.read_pdf_tables(pdf_filename, pages=pages,  make_NaN_blank=make_NaN_blank,
                                   read_many_tables_per_page=read_many_tables_per_page)
        expected_table_count = len(tables_to_tile)
        are_tables_ok = True
        if expected_table_count != len(dfs):
            self.logger.warning(f'There are {len(tables_to_tile)} tables to read, but read in {len(dfs)}')
            if len(dfs) < expected_table_count:
                self.logger.warning(f'Returning no tables')
                are_tables_ok = False
                return None
            else:
                self.logger.warning(f'Will use the first {expected_table_count} tables.')
        for i in tables_to_tile:
            if len(dfs[i]):
                self.logger.debug(f'Table {i} has {len(dfs[i])} records.')
            else:
                self.logger.warning(f'Table {i} is empty.')
                are_tables_ok = False

        table_layout = self._cu.layout(rows, cols,
                                       tile_by_rows)  # TODO: Change table_layout to work with tables_to_tile

        row_dfs = []
        for i, row in enumerate(table_layout):
            self.logger.debug(f'row {i} contains: {row}')
            this_row_df = self._pu.join_dfs_by_column([dfs[x] for x in row])
            row_dfs.append(this_row_df)
        pass # TODO Continue here

    def summarize_pdf_tables(self, pdf_filename: str, pages: Union[Ints, str] = 'all', make_NaN_blank: bool = True,
                        read_many_tables_per_page:bool=False, how_many_summarized: int = 10) -> Strings:
        """
        Summarize the tables on the given pages, and return the first how_many_summarized lines of each table.
        :param pdf_filename: Filename to scan
        :param pages: Default: 'all'. Could be a str like "1,2,3" or a list like [1,2,3].
        :param make_NaN_blank: True to make NaN values blank (default). False to leave as NaN.
        :param read_many_tables_per_page:
        :return:
        """
        ans = []
        dfs = self.read_pdf_tables(pdf_filename, pages, make_NaN_blank, read_many_tables_per_page)
        for i, df in enumerate(dfs):
            self.logger.info(self._su.fill_string(my_str=f'Table {i}'))
            df_head = self._pu.head_as_string(df, how_many_rows=7)
            self.logger.debug(df_head)
            ans.append(df_head)
        return ans