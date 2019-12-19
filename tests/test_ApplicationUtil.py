import unittest
from ApplicationUtil import ApplicationUtil
import platform
from FileUtil import FileUtil

"""
Class Test_ApplicationUtil last updated 7Dec19.
Interesting Python facets:
* Class under test is within this Test
* This class shows how to use the super().__init__ with an argument.

"""

class MyApplication(ApplicationUtil):
    def __init__(self, yaml_file:str):
        super().__init__(yaml_file)
    

class Test_ApplicationUtil(unittest.TestCase):
    fu = FileUtil()
    yaml_file = 'test.yaml'
    path = r'c:\temp' if platform.system() == 'Windows' else r'\tmp'
    qual_path = fu.qualified_path(path, yaml_file)
    yaml_dict = {'inputFile':qual_path, 'maxLines': 1000, 'vendorDict': {'SAN': 'Account number','Contact_fn': 'Contact first name'}}
        #,'Address': 'Address 1','Phone': 'Work phone','Name': 'Vendor name','Contact_Ln': 'Contact last name','Addr_2': 'Address 2','Ext': 'Mobile','email': 'E-Mail 1','City': 'City','St': 'State','Zip': 'Zip','Fax': 'Fax'}}

    @classmethod
    def setUpClass(cls) -> None:
        fu = FileUtil()
        fu.dump_yaml(Test_ApplicationUtil.qual_path, Test_ApplicationUtil.yaml_dict)

    def setUp(self) -> None:
        self.app = MyApplication(self.qual_path)

    def test_logger(self):
        self.app.logger.debug('log message from test_logger')


if __name__ == '__main__':
    unittest.main()
