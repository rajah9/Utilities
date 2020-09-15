import pandas as pd
import numpy as np
from typing import Callable, List, Union
import logging
from LogitUtil import logit
from FileUtil import FileUtil
from scipy.stats import linregress
from io import StringIO, TextIOWrapper, BufferedWriter

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

Strings = List[str]
Bools = List[bool]
Dataframes = List[pd.DataFrame]

"""
Interesting Python features:
* Uses a parameter dictionary to send arguments to a member.
* Uses typing to Union a string or a list.
* In replace_col_with_scalar, uses two different methods to replace a masked array with a scalar.
* In replace_col_using_func, I pass in a function to be executed by it.
* In stats, provided an independent way to find slope, intercept, and r.
* In the init, set the display options to show head() better.
* Uses a generator / yield function to generate column names
Rough outline:
0. Import Libraries
  a. from PandasUtil import PandasUtil
  b. pu = PandasUtil()
1. Import Dataset
  a. pu = PandasUtil()
  b. df = pu.read_df_from_csv(csv_file_name=r'C:\\Users\\Owner\\Documents\\Udemy\\ML-Classification-Package\\ML Classification Package\\7. Naive Bayes\emails.csv', sep=',')
  c. pu.get_rowCount_colCount(df)
  d. df.head(10)
  e. stats_df = pu.get_quartiles(df)
  f. pu.get_basic_data_analysis(df)
2. Visualize data 
  a. (see PlotUtil.py)
  b. count_df = pu.count_by_column(df, 'spam')
  c. count_df.head()
3. (see DataScienceUtil)
4. Training the model
  a. target_col = 'spam'
  b. all_cols = pu.get_df_headers(df)
  c. df_X = pu.drop_col(df, columns=target_col, is_in_place=False)
  d. df_y = pu.drop_col_keeping(df, cols_to_keep=target_col, is_in_place=False)
  e. X = pu.convert_dataframe_to_matrix(df_X)
  f. y = pu.convert_dataframe_to_vector(df_y)
"""


def generate_col_names(prefix: str) -> str:
    """
    Generator for col00 through col99.
    Invoke it like this:
      gen = generate_col_names('pfx')
      for i in range(3):
        print next(gen)  # prints pfx00, pfx01, pfx02.

    :return:
    """
    nums = range(100)  # Only 0 .. 99
    for i in nums:
        yield f'{prefix}{i:02d}'

class PandasUtil:
    _EMPTY_DF = pd.DataFrame()

    def __init__(self):
        self.filename = None
        self.worksheetName = None
        self._df = None
        self._fu = FileUtil()
        # make the df display look better: https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 800)

    # Getters and setters for filename, worksheetname, and df
    @property
    def filename(self):
        return self._filename
    # Setter for filename.
    @filename.setter
    def filename(self, fn:str):
        self._filename = fn

    @property
    def worksheetName(self):
        return self._worksheetName
    @worksheetName.setter
    def worksheetName(self, wks:str):
        self._worksheetName = wks

    @property
    def df(self):
        return self._df
    @df.setter
    def df(self, myDf:pd.DataFrame):
        self._df = myDf

    @classmethod
    def empty_df(cls) -> pd.DataFrame:
        return pd.DataFrame()

    def pandas_version(self):
        """
        Return the panas version as three ints
        :return: maj, minor, sub
        """
        v = pd.__version__
        majMinSub = [int(x) for x in v.split('.')]
        return majMinSub[0], majMinSub[1], majMinSub[2]

    def write_df_to_excel(self, df:pd.DataFrame=None, excelFileName:str=None, excelWorksheet:str=None, write_index=False) -> bool:
        """
        Write the given df to the excel file name and worksheet (unless
        they have already been provided and then are optional).
        Caller is responsible to catch any I/O errors.
        :param df:
        :param excelFileName:
        :param excelWorksheet:
        :return: True if Excel file written, False if df is empty.
        """
        if not df.empty:
            self._df = df
        else:
            logger.warning('Empty dataframe will not be written.')
            return False
        fn = excelFileName or self.filename
        wks = excelWorksheet or self.worksheetname
        writer = pd.ExcelWriter(fn)
        self._df.to_excel(writer, wks, index=write_index)
        writer.save()
        logger.debug(f'Successfully wrote to {fn}.')
        return True

    def write_df_to_csv(self, df:pd.DataFrame=None, csv_file_name:str=None, write_header:bool=True, write_index:bool=False, enc:str= 'utf-8') -> bool:
        """
        Write the given df to the file name and worksheet (unless
        they have already been provided and then are optional).
        Caller is responsible to catch any I/O errors.
        :param df:
        :param csv_file_name:
        :param write_header:
        :param write_index:
        :param enc:
        :return: True if Excel file written, False if df is empty.
        """
        if not df.empty:
            self._df = df
        else:
            logger.warning('Empty dataframe will not be written.')
            return False
        df.to_csv(csv_file_name, header=write_header, index=write_index, encoding=enc)
        logger.debug(f'Successfully wrote to {csv_file_name}.')
        return True

    def read_df_from_excel(self,excelFileName:str=None, excelWorksheet:str='Sheet1', header:int=0, index_col=None) -> pd.DataFrame:
        """
        Read an Excel file.
        :param excelFileName:
        :param excelWorksheet:
        :param header: 0-offset location of header (0=row 1 in Excel)
        :param index_col:
        :return: dataframe result
        """
        param_dict = {'header': header}
        if excelFileName:
            self.filename = excelFileName
        logger.debug(f'Will read from the Excel file: {self.filename}.')
        param_dict['io'] = self.filename
        if self._fu.file_exists(self.filename):
            if excelWorksheet:
                self.worksheetName = excelWorksheet
            wks = self.worksheetName
            major, minor, _ = self.pandas_version()
            logger.debug(f'Will read from the worksheet: {wks}. Pandas minor version is {minor}.')
            if wks not in self.get_worksheets(excelFileName):
                logger.warning(f'Cannot find Excel worksheet: {self.worksheetName}. Returning empty df.')
                return PandasUtil.empty_df()
            if ((major == 0) & (minor > 21)) | (major >= 1):
                param_dict['sheet_name'] = wks
            else:
                param_dict['sheetname'] = wks
            if index_col >= 0:
                param_dict['index_col'] = index_col
            self._df = pd.read_excel(**param_dict)
            logger.debug(f'Read in {len(self.df)} records.')
            return self._df
        else:
            logger.error(f'Cannot find Excel file: {self.filename}. Returning empty df.')
            return PandasUtil.empty_df()

    def read_df_from_csv(self, csv_file_name:str=None, header:int=0, enc:str= 'utf-8', index_col:int=None, sep:str = None) -> pd.DataFrame:
        """
        Write the given df to the file name and worksheet (unless
        they have already been provided and then are optional).
        :param df:
        :param csv_file_name:
        :param header: Where the headers live (0 means first line of the file)
        :param enc: try 'latin-1' or 'ISO-8859-1' if you are getting encoding errors
        :return:
        """
        param_dict = {'filepath_or_buffer': csv_file_name, 'header': header, 'encoding':enc,}
        if sep:
            param_dict['sep'] = sep
        if index_col is not None:
            param_dict['index_col'] = index_col
        ans = pd.read_csv(**param_dict)
        return ans


    def get_df_headers(self, df:pd.DataFrame=_EMPTY_DF) -> list:
        """
        Get a list of the headers.
        :param df:
        :param self:
        :return: list of headers
        """
        if not self.is_empty(df):
            self.df = df
            return list(self.df.columns)
        else:
            logger.warning('df is empty. Returning None for headers')
            return None

    def get_rowCount_colCount(self, df:pd.DataFrame):
        """
        Return the row and column_name count of the df.
        :param df:
        :return: row count, col count
        """
        rows, cols = df.shape
        logger.debug(f'df has {rows} rows and {cols} columns.')
        return rows, cols

    def get_basic_data_analysis(self, df:pd.DataFrame) -> str:
        buffer = StringIO()
        df.info(buf=buffer)
        ans = buffer.getvalue()
        logger.info(f'info:\n{ans}')
        return ans

    def get_quartiles(self, df:pd.DataFrame, percentiles: list = [.25, .50, .75]) -> pd.DataFrame:
        """
        Return basic statistics about the dataframe.
        :param df:
        :param percentiles: list of %-tiles as fractions between 0 and 1, e.g. [.2, .4, .6, .8] for quintiles
        :return: basic description df
        """
        ans = df.describe(percentiles=percentiles)
        logger.info(f'info:\n{ans.head(10)}')
        return ans

    @logit(showRetVal=True)
    def get_worksheets(self, excelFileName=None):
        if excelFileName:
            self.filename = excelFileName
        fu = FileUtil()
        if fu.file_exists(self.filename):
            xl = pd.ExcelFile(self.filename)
            return xl.sheet_names
        else:
            logger.error(f'Cannot find Excel file {self.filename}.')
            return None

    def duplicate_rows(self, df:pd.DataFrame, fieldList:list=None, keep:str='first') -> pd.DataFrame:
        """
        Return a dataframe with the duplicates as specified by the columns in fieldList.
        If fieldList is missing or None, then return the exactly duplicated rows.
        :param df: dataframe to scan for duplicates
        :param fieldList: fields in df to examine for duplicates.
        :param keep: 'first' or 'last' to keep the first dupe or the last.
        :return: df of the duplicates
        """
        if fieldList:
            ans = df[df.duplicated(fieldList, keep=keep)]
        else:
            ans = df[df.duplicated(keep=keep)]
        return ans


    def drop_duplicates(self, df:pd.DataFrame, fieldList:list=None, keep:str='first') -> pd.DataFrame:
        """
        Drop the duplicates as specified by the columns in fieldList.
        If fieldList is missing or None, then return the exactly duplicated rows.
        :param df: dataframe to scan for duplicates
        :param fieldList: fields in df to examine for duplicates.
        :param keep: 'first' or 'last' to keep the first dupe or the last.
        :return: df without the duplicates
        """
        param_dict = {'keep': keep, 'inplace': False}
        if fieldList:
            param_dict['subset'] = fieldList
        return df.drop_duplicates(**param_dict)

    def convert_dict_to_dataframe(self, list_of_dicts: list) -> pd.DataFrame:
        """
        Convert a list of dictionaries to a dataframe.
        :param list_of_dicts:
        :return:
        """
        return pd.DataFrame(list_of_dicts)

    def convert_list_to_dataframe(self, lists: list, column_names: List = None) -> pd.DataFrame:
        """
        Convert a list of lists to a dataframe. If provided, add the column names. If not, provide default col names.
        :param lists: list of objects
        :param column_names: Column names to use. Defaults to col00, col01, col22, .. col99
        :return:
        """
        if column_names:
            return pd.DataFrame(data=lists, columns=column_names)
        # Use the default column names: col00, col01...
        ans = pd.DataFrame(data=lists)
        self.replace_col_names_by_pattern(ans)
        return ans

    def convert_matrix_to_dataframe(self, lists: list) -> pd.DataFrame:
        """
        convert a list of lists to a dataframe.
        :param lists:
        :return:
        """
        return pd.DataFrame(data=lists)

    def convert_dataframe_to_matrix(self, df:pd.DataFrame) -> np.ndarray:
        """
        Convert all of the values to a numpy ndarray.

        :param df:
        :return:
        """
        return df.to_numpy()

    def convert_dataframe_to_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert the dataframe to a numpy vector.
        :param df:
        :return:
        """
        cols = self.get_df_headers(df)
        if len(cols) == 1:
            return df.to_numpy().reshape(-1,)
        logger.warning(f'Dataframe should have exactly one column, but contains {len(cols)}. Returning None.')
        return None

    def without_null_rows(self, df:pd.DataFrame, column_name:str) -> pd.DataFrame:
        """
        Return a DataFrame without the rows that are null in the given column_name.
        :param df: source DataFrame
        :param column_name: Column name to remove.
        :return: new DataFrame
        """
        try:
            mask = pd.notnull(df[column_name])
            return df[mask]
        except KeyError:
            logger.error(f'Unable to find column_name name: {column_name}. Returning empty df.')
            return PandasUtil.empty_df()

    def select(self, df:pd.DataFrame, column_name:str, match_me:Union[str, int]) -> pd.DataFrame:
        """
        Return a DataFrame that selects on the column_name that is equal to match_me.
        Similar to a SELECT * WHERE clause in SQL.
        :param df:
        :param column_name:
        :param match_me:
        :return: df with the column_name matching the selected clause (possibly empty)
        """
        return df.loc[df[column_name] == match_me]

    def mask_blanks(self, df:pd.DataFrame, column_name:str) -> list:
        """
        Return a boolean list with a True in the rows that have a blank column_name.
        :param df:
        :param column_name:
        :return:
        """
        # ans = df.loc[df[column_name] == '']
        ans = df[column_name] == ''
        return ans

    def select_blanks(self, df:pd.DataFrame, column_name:str) -> list:
        return df[self.mask_blanks(df, column_name)]

    def mask_non_blanks(self, df:pd.DataFrame, column_name:str) -> list:
        """
        Return a boolean list with a True in the rows that have a nonblank column_name.
        :param df:
        :param column_name:
        :return:
        """
        blanks = self.mask_blanks(df, column_name)
        non_blanks_mask = [not x for x in blanks]
        return non_blanks_mask

    def select_non_blanks(self, df: pd.DataFrame, column_name: str) -> list:
        return df[self.mask_non_blanks(df, column_name)]

    def unique_values(self, df:pd.DataFrame, column_name:str) -> list:
        """
        Return a list of the unique values in column_name.
        :param df:
        :param column_name:
        :return:
        """
        return self.drop_duplicates(df=df[column_name]).tolist()

    def count_by_column(self, df:pd.DataFrame, column_name:str=None) -> pd.DataFrame:
        """
        Return a count by value of the given column.
        :param df:
        :param column_name:
        :return:
        """
        return df[column_name].value_counts()

    def add_new_col_with_func(self, df:pd.DataFrame, column_name:str, func: Callable[[], list]) -> pd.DataFrame:
        """
        Call the func with no args to assign a new column_name to the dataframe.
        func should return a list comprehension.
        Here's an example of what the function should do.
            def my_func(self) -> list:
                df = self.pu.df
                col_of_interest = df['number']
                return [self.my_f(x) for x in col_of_interest]

        It gets called with:
            df = self.pu.add_new_col_with_func(df, 'new_col_name', self.my_func)

        :param df:
        :param column_name:
        :param func: func (usually no args)
        :return:
        """
        self.df = df
        df[column_name] = func()
        return df

    def add_new_col_from_array(self, df:pd.DataFrame, column_name:str, new_col:np.array) -> pd.DataFrame:
        """
        Use the values in new_col to create a new column.
        Limitations: this is not as sophisticated as https://stackoverflow.com/questions/12555323/adding-new-column-to-existing-dataframe-in-python-pandas .
        The length of new_col must be the same as the length of df.
        :param df:
        :param column_name:
        :param new_col: If this really is a Series, it will try to match indexes with the existing df (probably a good thing).
        :return:
        """
        df[column_name] = new_col
        return df


    def mark_rows_by_func(self, df:pd.DataFrame, column_name:str, func: Callable[[], list]) -> Bools:
        """
        Return a list of bools depending on the func.
        Here's a func (which takes a list as a parameter):
            def is_adult(self, age:list):
                return age >= 21
        Here's how to invoke it:
            mark = self.pu.mark_rows_by_func(df, 'Age', self.is_adult)

        :param df: dataframe under scrutiny
        :param column_name: name of the column_name
        :param func:   function that is to be invoked. Takes a list and returns a list of booleans.
        :return:
        """
        mask = func(df[column_name])
        return mask

    def mark_rows_by_criterion(self, df:pd.DataFrame, column_name:str, criterion:Union[str,int,float]) -> Bools:
        """
        Return a list of bools when column_name meets the criterion.
        :param df:
        :param column_name:
        :param criterion:
        :return:
        """
        mask = df[column_name] == criterion
        return mask

    def mark_isnull(self, df:pd.DataFrame, column_name:str) -> Bools:
        mask = df[column_name].isnull()
        return mask

    def masked_df(self, df:pd.DataFrame, mask:Bools, invert_mask:bool=False):
        if not invert_mask:
            return df[mask]
        else:
            my_mask = [not x for x in mask]
            return df[my_mask]

    def set_index(self, df:pd.DataFrame, columns: Union[Strings, str], is_in_place:bool = True) -> pd.DataFrame:
        """
        Set the index of df.

        :param df: Dataframe under scrutiny.
        :param columns: Can be a str (=single column_name) or a List of strings.
        :param is_in_place: True to add the index in place / False to create a new df
        :return: df or None (if is_in_place is true)
        """
        return df.set_index(columns, inplace=is_in_place)

    def reset_index(self, df:pd.DataFrame, is_in_place:bool = True, is_dropped:bool = False) -> pd.DataFrame:
        """
        Reset the index.
        :param df:
        :param is_in_place:
        :param is_dropped:
        :return:
        """
        return df.reset_index(drop=is_dropped, inplace=is_in_place)

    def drop_index(self, df:pd.DataFrame, is_in_place:bool = True) -> pd.DataFrame:
        """
        Drop the index
        :param df:
        :param is_in_place:
        :param is_dropped:
        :return:
        """
        return self.reset_index(df=df, is_in_place=is_in_place, is_dropped=True)

    def drop_col(self, df:pd.DataFrame, columns: Union[Strings, str], is_in_place:bool = True) -> pd.DataFrame:
        """
        Drop the given column_name.
        :param df:
        :param columns: Can be a str (=single column_name) or a List of strings.
        :param is_in_place: if true, column_name is dropped from df in place. Otherwise, a new df is returned.
        :return: None if is_in_place is True. Else df with the column_name dropped.
        """
        major, minor, _ = self.pandas_version()
        if (major == 0) & (minor < 21):
            logger.warning(f'Unable to drop column, as Pandas version is {minor}. Returning unchanged df.')
            return df

        return df.drop(columns=columns, inplace=is_in_place)

    @logit()
    def drop_col_keeping(self, df:pd.DataFrame, cols_to_keep: Union[Strings, str], is_in_place:bool = True) -> pd.DataFrame:
        """
        Keep the given columns and drop the rest.
        :param df:
        :param cols_to_keep:
        :param is_in_place:
        :return:
        """
        headers_to_drop = self.get_df_headers(df)
        logger.debug(f'I have these headers: {headers_to_drop}. But I will keep {cols_to_keep}')
        exceptions = cols_to_keep
        if isinstance(cols_to_keep, str):
            exceptions = [cols_to_keep]
        for col in exceptions:
            headers_to_drop.remove(col)
        return self.drop_col(df=df, columns=headers_to_drop, is_in_place=is_in_place)

    def drop_row_by_criterion(self, df:pd.DataFrame, column_name: str, criterion: str, is_in_place:bool = True) -> pd.DataFrame:
        """
        Drop the rows that have criterion in the given column.
        :param df:
        :param column_name:
        :param criterion:
        :param is_in_place:
        :return:
        """
        ans = df[df[column_name] != criterion]
        if is_in_place:
            df = ans
        else:
            return ans


    def reorder_cols(self, df:pd.DataFrame, columns: Strings) -> pd.DataFrame:
        """
        Using the columns, return a new df.
        :param df:
        :param columns: list of strings, like ['colD', 'colA', 'colB', 'colC']
        :return:
        """
        return df[columns]

    def replace_col(self, df:pd.DataFrame, column: str, replace_dict: dict) -> pd.DataFrame:
        """
        Replace the values of column_name using replace_dict.
        :param df:
        :param column:
        :param replace_dict: {'origA':'replA', 'origB':'replB'}
        :return: df with column_name replaced
        """
        try:
            df[column] = df[column].map(replace_dict)
        except KeyError:
            logger.warning(f'Value found outside of: {replace_dict.keys()} or column_name {column} not found. Returning empty df.')
            return self.empty_df()
        return df

    def replace_col_using_func(self, df:pd.DataFrame, column_name: str, func: Callable[[], list]) -> pd.DataFrame:
        """
        Replace the column contents by each element's value, as determined by func.
        :param df: Dataframe under scrutiny.
        :param column_name: (single column_name) name
        :param func: Function operates on whatever element it is presented, and returns the changed element.
        :return: df
        """
        df[column_name] = df[column_name].apply(func)
        return df

    def replace_col_using_mult_cols(self, df:pd.DataFrame, column_to_replace: str, cols: Strings, func: Callable[[], list]) -> pd.DataFrame:
        """

        :param df: Dataframe under scrutiny.
        :param column_to_replace: (single column_name) name
        :param cols: list of columns used for the following func
        :param func:
        :return: df with replaced column
        """
        df[column_to_replace] = df[cols].apply(func, axis=1)
        return df

    def replace_col_with_scalar(self, df:pd.DataFrame, column_name: str, replace_with: Union[str, int], mask: Bools=None) -> pd.DataFrame:
        """
        Replace the all column_name with replace_with. If a mask of bools is used, only replace those elements with a True.
        Helpful reference at https://kanoki.org/2019/07/17/pandas-how-to-replace-values-based-on-conditions/
        :param df:
        :param column_name:
        :param replace_with:
        :param mask:
        :return:
        """
        if mask is None:
            df[column_name] = replace_with
        elif isinstance(mask, pd.Series):
            df[column_name].mask(mask.tolist(), replace_with, inplace=True)
        elif isinstance(mask, list) :
            # df[column_name].mask(mask, replace_with, inplace=True) # Method 1 and works
            df.loc[mask, column_name] = replace_with                 # Method 2 at kanoki.
        else:
            logger.warning(f'mask must be None, a series, or a list, but it is: {type(mask)}')
            return self.empty_df()

    def join_two_dfs_on_index(self, df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
        """
        return a column-wise join of these two dataframes on their mutual index.
        :param df1:
        :param df2:
        :return:
        """
        return pd.concat([df1, df2], axis=1, ignore_index=False)

    def join_dfs_by_column(self, dfs: Dataframes) -> pd.DataFrame:
        """
        Return a column-wise join of these dataframes.
        :param dfs:
        :return:
        """
        return pd.concat(dfs, axis='columns')

    def join_dfs_by_row(self, dfs: Dataframes) -> pd.DataFrame:
        """
        Return a row-wise join of these dataframes.
        :param dfs:
        :return:
        """
        return pd.concat(dfs, axis='rows', ignore_index=True)

    def dummy_var_df(self, df:pd.DataFrame, columns: Union[Strings, str], drop_first:bool=True) -> pd.DataFrame:
        """
        Do a one-hot encoding.
        Create a dummy variable based on the given column.
        :param df:
        :param columns: a single column name or a list of column names.
        :return:
        """
        if isinstance(columns, str):
            my_columns = [columns]
        else:
            my_columns = columns
        df = pd.get_dummies(data=df, columns=my_columns, drop_first=drop_first)
        return df

    def replace_col_names(self, df:pd.DataFrame, replace_dict: dict, is_in_place:bool = True) -> pd.DataFrame:
        """
        :param replace_dict: {'origColA':'replColA', 'origColB':'replColB'}

        """
        return df.rename(columns=replace_dict, inplace=is_in_place)


    def replace_col_names_by_pattern(self, df: pd.DataFrame, prefix: str = "col", is_in_place: bool = True) -> pd.DataFrame:
        """
        Replace the column names with col1, col2....
        :param df:
        :param prefix: string prefix, such as "col"
        :param is_in_place:
        :return:
        """
        cur_names = self.get_df_headers(df)
        gen = generate_col_names(prefix)
        replacement_dict = {k: next(gen) for k in cur_names}
        return self.replace_col_names(df, replacement_dict, is_in_place)


    def coerce_to_string(self, df:pd.DataFrame, columns: Union[Strings, str]) -> pd.DataFrame:
        """
        Coerce the given column_name name to a string.
        :param df:
        :param column_name:
        :return: new df with column_name coerced to str.
        """
        if isinstance(columns, str):
            # Make the single str columns into a list with just that one element.
            cols_as_list = [columns]
        else:
            cols_as_list = columns
        for col in cols_as_list:
            df[col] = df[col].apply(str)
        return df

    def coerce_to_numeric(self, df:pd.DataFrame, columns: Union[Strings, str]) -> pd.DataFrame:
        """
        Coerce the given column_name name to ints or floats.
        :param df:
        :param column_name:
        :return: new df with column_name coerced to a numeric.
        """
        if isinstance(columns, str):
            # Make the single str columns into a list with just that one element.
            cols_as_list = [columns]
        else:
            cols_as_list = columns
        df[cols_as_list] = df[cols_as_list].apply(pd.to_numeric)
        return df

    def round(self, df:pd.DataFrame, rounding_dict:dict) -> pd.DataFrame:
        """
        Round the columns given in rounding_dict to the given number of decimal places.
        Unexpected result found in testing: python function round(4.55, 2) yields 4.5 BUT this function returns 4.6
        :param df:
        :param rounding_dict: {'A': 2, 'B':3}
        :return: df rounded to the specified number of places.
        """
        return df.round(rounding_dict)

    def replace_vals(self, df:pd.DataFrame, replace_me:str, new_val:str, is_in_place:bool = True) -> pd.DataFrame:
        """
        Replace the values of replace_me with the new_val.

        :param df: Dataframe under scrutiny.
        :param
        :param is_in_place: True to replace values in place / False to create a new df
        :return: df or None (if is_in_place is true)
        """
        return df.replace(to_replace=replace_me, value=new_val, inplace=is_in_place)

    def replace_vals_by_mask(self, df:pd.DataFrame, mask:Bools, col_to_change:str, new_val:Union[str, int, float]):
        """
        Replace the values in the col_to_change with the new_val
        :param df:
        :param mask:
        :param col_to_change: Column Name whose rows you want to change
        :param new_val:
        :return: the changed df (also changed in place)
        """
        ans = df.loc[mask, col_to_change] = new_val
        return ans

    def is_empty(self, df:pd.DataFrame) -> bool:
        """
        Return true if the df is empty.
        :param df: Dataframe to inspect
        :return: True IFF it is empty
        """
        return df.empty

    def aggregates(self, df:pd.DataFrame, group_by:Strings, col:str) -> pd.DataFrame:
        """
        Return the average, min, max, and sum of the dataframe when grouped by the given strings.
        Reference: https://jamesrledoux.com/code/group-by-aggregate-pandas .
        :param df:
        :param group_by:
        :return:
        """
        grouped_multiple = df.groupby(group_by).agg({col: ['mean', 'min', 'max', 'sum']})
        grouped_multiple.columns = ['mean', 'min', 'max', 'sum']
        self.reset_index(grouped_multiple, is_in_place=True)
        return grouped_multiple

    def stats(self, df: pd.DataFrame, xlabel_col_name: str, ylabel_col_name: str):
        """
        Calculate the main statistics.
        :param df: dataframe under scrutiny
        :param xlabel_col_name: x column label
        :param ylabel_col_name: y column label
        :return: slope, intercept, and r (correlation)
        """
        slope, intercept, r, p, epsilon = linregress(df[xlabel_col_name], df[ylabel_col_name])
        logger.info('Main equation: y = %.3f x + %.3f' % (slope, intercept))
        logger.info('r^2 = %.4f' % (r * r))
        logger.info('p = %.4f' % (p))
        logger.info('std err: %.4f' % (epsilon))
        return slope, intercept, r

    def head(self, df: pd.DataFrame, how_many_rows:int=10) -> pd.DataFrame:
        """
        Return the first how_many_rows. This works well if called as the last line of an immediate, as in:
          pu.head(df)
        :param df:
        :param how_many_rows:
        :return:
        """
        self.df = df
        return self.df.head(how_many_rows)

    def head_as_string(self, df: pd.DataFrame, how_many_rows:int=10) -> str:
        """
        Return the first how_many_rows as a string, separated by \n.
        :param df:
        :param how_many_rows:
        :return:
        """
        ans = str(self.head(df, how_many_rows))
        logger.debug(f'First {how_many_rows} are:\n{ans}')
        return ans

    def tail_as_string(self, df: pd.DataFrame, how_many_rows:int=10) -> str:
        """
        Return the last how_many_rows as a string, separated by \n.
        :param df:
        :param how_many_rows:
        :return:
        """
        ans = str(self.tail(df, how_many_rows))
        logger.debug(f'Last {how_many_rows} are:\n{ans}')
        return ans


    def tail(self, df: pd.DataFrame, how_many_rows:int=10) -> pd.DataFrame:
        """
        Return the last how_many_rows. This works well if called as the last line of an immediate, as in:
          pu.tail(df)
        :param df:
        :param how_many_rows:
        :return:
        """
        self.df = df
        return self.df.tail(how_many_rows)

    def sort(self, df: pd.DataFrame, columns: Union[Strings, str], is_in_place:bool = True, is_asc: bool = True):
        """
        Sort the given dataFrame by the given column(s).
        :param df:
        :param columns:
        :param is_in_place:
        :param is_asc:
        :return:
        """
        return df.sort_values(columns, ascending=is_asc, inplace=is_in_place, kind='quicksort', na_position='last')

class DataFrameSplit():
    """
    Class to implement an iterator to divide a dataframe.
    """
    def __init__(self, my_df:pd.DataFrame, interval:int = 500):
        logger.debug(f'Initializing with df of length {len(my_df)} and interval of {interval}.')
        self.df = my_df
        self.interval = interval
        self.max = len(my_df)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        logger.debug(f'entering n={self.n}, max={self.max}, interval={self.interval}')
        if self.n >= self.max:
            logger.debug('Stopping.')
            raise StopIteration
        if (self.n + self.interval) <= self.max:
            logger.debug(f'Continuing. n={self.n} + interval={self.interval} <= {self.max}')
            end = self.n + self.interval - 1
        else:
            logger.debug(f'Ending soon. n={self.n}, max={self.max}')
            end = self.max
        ans = self.df[self.n:end]
        self.n = end
        return ans
