import pandas as pd
from typing import Callable, List, Union
import logging
from LogitUtil import logit
from FileUtil import FileUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

Strings = List[str]
Bools = List[bool]

"""
Interesting Python features:
* Uses a parameter dictionary to send arguments to a member.
* Uses typing to Union a string or a list.
"""

class PandasUtil:
    _EMPTY_DF = pd.DataFrame()

    def __init__(self):
        self.filename = None
        self.worksheetName = None
        self._df = None
        self._fu = FileUtil()

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

    def read_df_from_excel(self,excelFileName:str=None, excelWorksheet:str='Sheet1', header:int=0, index_col=None):
        param_dict = {'header': header}
        if excelFileName:
            self.filename = excelFileName
        logger.debug(f'Will read from the Excel file: {self.filename}.')
        param_dict['io'] = self.filename
        if self._fu.file_exists(self.filename):
            if excelWorksheet:
                self.worksheetName = excelWorksheet
            wks = self.worksheetName
            _, minor, _ = self.pandas_version()
            logger.debug(f'Will read from the worksheet: {wks}. Pandas minor version is {minor}.')
            if wks not in self.get_worksheets(excelFileName):
                logger.warning(f'Cannot find Excel worksheet: {self.worksheetName}. Returning empty df.')
                return PandasUtil.empty_df()
            if minor > 21:
                param_dict['sheet_name'] = wks
            else:
                param_dict['sheetname'] = wks
            if index_col:
                param_dict['index_col'] = index_col
            self._df = pd.read_excel(**param_dict)
            logger.debug(f'Read in {len(self.df)} records.')
            return self._df
        else:
            logger.error(f'Cannot find Excel file: {self.filename}. Returning empty df.')
            return PandasUtil.empty_df()

    def read_df_from_csv(self, csv_file_name:str=None, header:int=0, enc:str= 'utf-8', index_col:int=None, separator:str = ',') -> pd.DataFrame:
        """
        Write the given df to the file name and worksheet (unless
        they have already been provided and then are optional).
        :param df:
        :param csv_file_name:
        :param header: Where the headers live (0 means first line of the file)
        :param enc:
        :return:
        """
        param_dict = {'filepath_or_buffer': csv_file_name, 'header': header, 'encoding': enc, 'sep': separator}
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
        return pd.DataFrame(list_of_dicts)

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

    def add_new_col(self, df:pd.DataFrame, column_name:str, func: Callable[[], list]) -> pd.DataFrame:
        """
        Call the func with no args to assign a new column_name to the dataframe.
        func should return a list comprehension.
        Here's an example of what the function should do.
            def my_func(self) -> list:
                df = self.pu.df
                col_of_interest = df['number']
                return [self.my_f(x) for x in col_of_interest]

        It gets called with:
            df = self.pu.add_new_col(df, 'new_col_name', self.my_func)

        :param df:
        :param column_name:
        :param func: func (usually no args)
        :return:
        """
        self.df = df
        df[column_name] = func()
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
        :param column_name: name of the column
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

    def drop_col(self, df:pd.DataFrame, columns: Union[Strings, str], is_in_place:bool = True) -> pd.DataFrame:
        """
        Drop the given column_name.
        :param df:
        :param columns: Can be a str (=single column_name) or a List of strings.
        :param is_in_place: if true, column_name is dropped from df in place. Otherwise, a new df is returned.
        :return: None if is_in_place is True. Else df with the column_name dropped.
        """
        return df.drop(columns=columns, inplace=is_in_place)

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


    def replace_col_using_func(self, df:pd.DataFrame, column: str, func: Callable[[], list]) -> pd.DataFrame:
        """
        Replace the colun contents by each element's value, as determined by func.
        :param df: Dataframe under scrutiny.
        :param column: (single column_name) name
        :param func: Function operates on whatever element it is presented, and returns the changed element.
        :return: df
        """
        df[column] = df[column].apply(func)
        return df

    def replace_col_with_scalar(self, df:pd.DataFrame, column_name: str, replace_with: Union[str, int], mask: Bools=None) -> pd.DataFrame:
        """
        Replace the all column_name with replace_with. If a mask of bools is used, only replace those elements with a True.
        :param df:
        :param column_name:
        :param replace_with:
        :param mask:
        :return:
        """
        if mask:
            df[column_name] = replace_with
        else:
<<<<<<< Updated upstream
            df[column_name] = replace_with
=======
            logger.warning(f'mask must be None, a series, or a list, but it is: {type(mask)}')
            return self.empty_df()

    def join_dfs_on_index(self, df1:pd.DataFrame, df2:pd.DataFrame) -> pd.DataFrame:
        """
        return the join of these two dataframes on their index.
        :param df1:
        :param df2:
        :return:
        """
        pass # TODO Return an error if they don't both have indexes
        return pd.concat([df1, df2], axis=1)

    def dummy_var_df(self, df:pd.DataFrame, columns: Union[Strings, str], drop_first:bool=True) -> pd.DataFrame:
        """
        create a dummy variable based on the given column
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

>>>>>>> Stashed changes

    def replace_col_names(self, df:pd.DataFrame, replace_dict: dict, is_in_place:bool = True) -> pd.DataFrame:
        """
        :param replace_dict: {'origColA':'replColA', 'origColB':'replColB'}

        """
        return df.rename(columns=replace_dict, inplace=is_in_place)

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

<<<<<<< Updated upstream
=======
    def stats(self, df: pd.DataFrame, xlabel_col_name: str, ylabel_col_name: str):
        """
        Calculate the main statistics
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

>>>>>>> Stashed changes
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
