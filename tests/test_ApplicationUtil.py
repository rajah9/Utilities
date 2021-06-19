import logging
import platform
import unittest

import pandas as pd
from pandas.util.testing import assert_frame_equal

from ApplicationUtil import ApplicationUtil
from FileUtil import FileUtil
from LogitUtil import logit
from PandasUtil import PandasUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Class Test_ApplicationUtil last updated 7Dec19.
Interesting Python facets:
* Class under test is within this Test
* This class shows how to use the super().__init__ with an argument.
* testing for an empty dictionary with bool(d) in test_missing_yaml

"""

class MyApplication(ApplicationUtil):
    def __init__(self, yaml_file:str):
        super().__init__(yaml_file)
    

class Test_ApplicationUtil(unittest.TestCase):
    fu = FileUtil()
    yaml_file = 'test.yaml'
    path = r'c:\temp' if platform.system() == 'Windows' else r'/tmp'
    qual_path = fu.qualified_path(path, yaml_file)
    excel_yaml_file = 'excel_' + yaml_file
    excel_qual_path = fu.qualified_path(path, excel_yaml_file)
    spreadsheet_name = 'test.xls'
    worksheet_name = 'Sheet1'

    yaml_dict = {'inputFile':qual_path, 'maxLines': 1000, 'vendorDict': {'SAN': 'Account number','Contact_fn': 'Contact first name'}}
        #,'Address': 'Address 1','Phone': 'Work phone','Name': 'Vendor name','Contact_Ln': 'Contact last name','Addr_2': 'Address 2','Ext': 'Mobile','email': 'E-Mail 1','City': 'City','St': 'State','Zip': 'Zip','Fax': 'Fax'}}

    @classmethod
    def setUpClass(cls) -> None:
        fu = FileUtil()
        fu.dump_yaml(Test_ApplicationUtil.qual_path, Test_ApplicationUtil.yaml_dict)

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if platform.system() == 'Windows' else r'/tmp'
        fu.delete_file(fu.qualified_path(path, cls.yaml_file))
        fu.delete_file(fu.qualified_path(path, cls.excel_qual_path))
        fu.delete_file(fu.qualified_path(path, cls.spreadsheet_name))

    def setUp(self) -> None:
        self.app = MyApplication(self.qual_path)

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

    def test_logger(self):
        self.app.logger.debug('log message from test_logger')

    @logit()
    def test_missing_yaml(self):
        qualified_path = self.fu.qualified_path(dirPath=self.path, filename='noSuchFile.yaml')
        app = MyApplication(qualified_path)
        self.assertDictEqual(app._d.asdict, {})

    @logit()
    def test_yaml_entry(self):
        # Test 1
        for key, value in self.yaml_dict.items():
            actual = self.app.yaml_entry(key)
            logging.debug(f'comparing key / value of {key} / {value} against {actual}')
            self.assertEqual(value, actual)
        # Test 2
        expected_log_message = 'Unable to find yaml key'
        with self.assertLogs(ApplicationUtil.__name__, level='DEBUG') as cm:
            actual = self.app.yaml_entry('No such key!!')
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
            self.assertEqual(actual, '')

    @logit()
    def test_load_df_from_excel(self):
        pu = PandasUtil()
        fu = FileUtil()
        df = self.my_test_df()
        fn = self.fu.qualified_path(self.path, self.spreadsheet_name)
        pu.write_df_to_excel(df=df, excelFileName=fn, excelWorksheet=self.worksheet_name, write_index=False)
        yaml_dict = {'inputFile':fn, 'worksheet': self.worksheet_name }
        fu.dump_yaml(Test_ApplicationUtil.excel_qual_path, yaml_dict)
        app = MyApplication(Test_ApplicationUtil.excel_qual_path)
        actual = app.load_df_from_excel(input_file_yaml_entry='inputFile', worksheet=self.worksheet_name)
        assert_frame_equal(df, actual)

    @logit()
    def test_write_excel(self):
        pu = PandasUtil()
        fu = FileUtil()
        df = self.my_test_df()
        fn = self.fu.qualified_path(self.path, self.spreadsheet_name)
        yaml_dict = {'outputFile':fn, 'worksheet': self.worksheet_name }
        fu.dump_yaml(Test_ApplicationUtil.excel_qual_path, yaml_dict)
        app = MyApplication(Test_ApplicationUtil.excel_qual_path)
        app.write_excel(df=df, output_file_yaml_entry='outputFile', worksheet=self.worksheet_name)

        actual = pu.read_df_from_excel(excelFileName=fn, excelWorksheet=self.worksheet_name)
        assert_frame_equal(df, actual)



if __name__ == '__main__':
    unittest.main()
