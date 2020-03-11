import sys
from pathlib import Path
import logging
from unittest import mock, TestCase, main
from ExecUtil import ExecUtil
from LogitUtil import logit
from FileUtil import FileUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* Uses mocking.
** with @mock.patch.object(ExecUtil, 'executing_file')
** does a mock chained call 
* generate_text_lines uses a return type of List[str].
* Has an inner function
"""

class Test_ExecUtil(TestCase):
    def setUp(self):
        self.eu = ExecUtil()
        self.mock_file = r'C:\mockpath\mockfile.py'
        self.mock_abspath = r'c:\abspath\mockfile.py'

    def test_which_platform(self):
        plat = "Windows" if "win" in sys.platform else "Linux"
        self.assertEqual(plat, self.eu.which_platform())

    @logit()
    @mock.patch.object(ExecUtil, 'executing_file')
    def test_exec_file_path(self, mock_obj):
        mock_obj.return_value = self.mock_file
        self.assertEqual(self.mock_file, self.eu.exec_file_path())

    @logit()
    @mock.patch('ExecUtil.Path')
    def test_parent_folder(self, mock_path):
        parent = r'c:\mock'
        path = parent + r'\sub'
        mock_path.return_value = Path(path)
        logger.debug(f'calling parent_folder with mock path of: {path}')
        actual = self.eu.parent_folder()
        logger.debug(f'got actual parent folder of: {actual}')
        self.assertEqual(parent, str(actual))

    @logit()
    @mock.patch('ExecUtil.Path')
    def test_parent_folder_at_root(self, mock_path):
        path = 'c:\\'
        mock_path.return_value = Path(path)
        logger.debug(f'calling parent_folder with mock path of: {path}')
        actual = self.eu.parent_folder()
        logger.debug(f'got actual parent folder of: {actual}')
        self.assertEqual(path, str(actual)) # "parent" of root is still the parent.

    @logit()
    @mock.patch('ExecUtil.abspath')
    @mock.patch.object(ExecUtil, 'executing_file')
    def test_exec_file_path_err(self, mock_obj, mock_abs):
        mock_abs.return_value = self.mock_abspath
        mock_obj.side_effect = NameError(mock_obj, 'mock name error')
        self.assertEqual(self.mock_abspath, self.eu.exec_file_path())  # This throws the NameError

    @logit()
    @mock.patch.object(ExecUtil, 'executing_file')
    def test_executing_directory(self, mock_obj):
        mock_obj.return_value = self.mock_file
        fu = FileUtil()
        path, _ = fu.split_qualified_path(self.mock_file)
        self.assertEqual(path, self.eu.executing_directory())

    def test_add_path(self):
        my_path = self.eu.executing_file()
        appended_paths = self.eu.add_path(my_path)
        self.assertTrue(my_path in appended_paths)

    @logit()
    @mock.patch('ExecUtil.Popen.communicate')
    def test_exec_os_cmd(self, mock_popen):
        # Need a chained call for resp = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, close_fds=True).communicate()[0]
        expected =  [
            ' Volume in drive C has no label.',
            ' Volume Serial Number is C419-12FD',
            ' Directory of C:\\Users\\Owner\\PycharmProjects\\Utilities\\tests',
            '10/17/2019  08:35 AM    <DIR>          .',
            '10/17/2019  08:35 AM    <DIR>          ..',
            '10/16/2019  04:46 PM             3,002 test_DateUtil.py',
            '10/17/2019  08:35 AM             2,111 test_ExecUtil.py',
            '10/17/2019  08:23 AM             9,911 test_FileUtil.py',
            '10/15/2019  12:05 PM             7,773 test_PandasUtil.py',
            '09/09/2019  12:38 PM             3,148 test_StringUtil.py',
            '               5 File(s)         25,945 bytes',
            '               2 Dir(s)  659,333,849,088 bytes free']
        expected_as_bytes = '\n'.join(expected).encode()
        mock_popen.return_value = [expected_as_bytes]
        actual = self.eu.exec_os_cmd('dir')
        self.assertEqual(expected, actual)

    @mock.patch.object(ExecUtil, 'exec_os_cmd')
    def test_calc_total_bytes(self, mock_obj):
        d = {'filea.txt':12345, 'fileb.txt':67890, 'filec.txt':1001001}
        lines = [ f'{v} {k}' for k, v in d.items() ]
        mock_obj.return_value = lines
        expected = sum(d.values())
        actual = self.eu.calc_total_bytes('mockpath')
        self.assertEqual(actual, expected)
        # inner function
        def test_parse_hadoop(self):
            ex = list(d.values())
            act = self.eu.parse_hadoop_ls(lines)
            self.assertEqual(ex, act)

    @logit()
    def test_which_architecture(self):
        actual = self.eu.which_architecture()
        self.assertTrue(actual in ['32bit', '64bit'])

if __name__ == '__main__':
    main()
