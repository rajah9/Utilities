import logging
from FileUtil import FileUtil
from YamlUtil import YamlUtil
from PandasUtil import PandasUtil
import pandas as pd

"""
This is a class that reads in a yaml file. It also provides some helpful, ubiquitous utilities:
* logging
* self.df and self.pu for pandas utilities
* self.fu for file utilities

"""

class ApplicationUtil:
    df = None
    _d = {}
    _tuple = None
    pu = PandasUtil()
    fu = FileUtil()

    def __init__(self, yaml_file:str):
        self.logger = self.init_logger()
        d = YamlUtil(yaml_file)
        self._tuple = d.asnamedtuple
        self._d = d
        self.logger.debug(f'Read in yaml file {yaml_file} with fields: {self._d.fields}')

    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # add fromatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)
        return self.logger

    def yaml_entry(self, yaml_entry:str) -> str:
        """
        Read the dictionary and return the value of the given key. Give a warning if the yaml_entry is missing and return a blank.
        :param yaml_entry:
        :return:
        """
        try:
            return self._d.asdict[yaml_entry]
        except KeyError:
            self.logger.warning(f'Unable to find yaml key: {yaml_entry}. Returning blank value.')
            return ''

    def load_df_from_excel(self, input_file_yaml_entry:str, worksheet:str='Sheet1'):
        input_file = self._d.asdict[input_file_yaml_entry]
        self.logger.debug(f'Reading {worksheet} file: {input_file}')
        if self.fu.file_exists(input_file):
            df = self.pu.read_df_from_excel(excelFileName=input_file, excelWorksheet=worksheet, header=0)
            self.pu.get_rowCount_colCount(df)
            return df
        else:
            self.logger.warning(f'Unable to find {worksheet} file: {input_file_yaml_entry}. Returning empty dataframe.')
            return self.pu.empty_df()

    def write_excel(self, df:pd.DataFrame, output_file_yaml_entry:str, worksheet:str) -> None:
        """
        Write the given dataframe to the file indicated by the dictionary entry (that was read in using the yaml file).
        :param df: DataFrame to write
        :param output_file_yaml_entry:
        :param worksheet:
        :return:
        """
        output_file = self.yaml_entry(output_file_yaml_entry)
        self.logger.debug(f'Writing {worksheet} file: {output_file}')
        self.pu.write_df_to_excel(df=df, excelFileName=output_file, excelWorksheet=worksheet)
