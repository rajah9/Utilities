import logging
import platform
from io import StringIO
from os import sep
from random import randrange
from typing import List
from unittest import TestCase, mock, main
from yaml import YAMLError
from DateUtil import DateUtil
from FileUtil import FileUtil
from ExecUtil import ExecUtil
from LogitUtil import logit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* Does a super init
* Uses tearDownClass (classmethod) to delete test files.
* Uses mocking.
** with mock.patch('FileUtil.getcwd', return_value=my_mock_dir):
** Note that the part within quotes is FileUtil and just getcwd (not os.cwd), because that's how I call it in FileUtil.
** also with @mock.patch decorator. Tests NameError and open errors.
** Uses self.assertLogs to ensure that an error message is logged in the calling routine.
** Has a nested mock.patch in test_get_files, which uses a local side effect function to yield 
**     different results (simulating a file or a directory).
* generate_text_lines uses a return type of List[str].
* Uses next((True for color in colors if search in color), False) to search for a substring within a list of strings.
"""


def mock_is_file(file_or_dir:str) -> bool:
    return '.txt' in file_or_dir

def mock_is_dir(file_or_dir: str) -> bool:
    return not mock_is_file(file_or_dir)


class Test_FileUtil(TestCase):
    path_no_drive = 'temp'
    fn = 'test.csv'
    yaml = 'example.yaml'
    text_fn = 'test.txt'

    def __init__(self, *args, **kwargs):
        super(Test_FileUtil, self).__init__(*args, **kwargs)
        self.path = r'c:\temp' if platform.system() == 'Windows' else r'\tmp'
        self._fu = FileUtil()
        self._du = DateUtil()
        self.features_dict = {'book': "Hitchhiker's Guide", 'characters': {'answer':42, 'name': 'Dent. Arthur Dent.'}}

    @classmethod
    def tearDownClass(cls) -> None:
        fu = FileUtil()
        path = r'c:\temp' if platform.system() == 'Windows' else r'\tmp'
        fu.delete_file(fu.qualified_path(path, cls.yaml))
        fu.delete_file(fu.qualified_path(path, cls.fn))
        fu.delete_file(fu.qualified_path(path, cls.text_fn))

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p

    def create_csv(self):
        lines = [',col1,col2',
                 '0,1,3',
                 '1,2,4',
                 ]
        filename = self._fu.qualified_path(self.path, self.fn)
        self._fu.write_text_file(filename, lines)
        logger.debug(f'create_csv to {self.path}{sep}{self.fn}.')

    def create_yaml(self, keys: list, vals: list):
        writeMe = []
        for i in range(len(keys)):
            writeMe.append(f'{keys[i]}: {vals[i]}')

        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        self._fu.write_text_file(filename=qualifiedPath, lines=writeMe)

    def generate_text_lines(self, how_many: int = 10) -> List[str]:
        return [f'Line {i}' for i in range(how_many)]

    def create_text_file(self, filename: str, how_many: int = 10):
        lines = self.generate_text_lines(how_many)
        self._fu.write_text_file(filename, lines)

    @logit()
    def test_dump_yaml(self):
        yaml_fn = self._fu.qualified_path(self.path, self.yaml)
        self._fu.dump_yaml(yaml_fn, self.features_dict)
        self.assertTrue(self._fu.file_exists(yaml_fn))
        actual = self._fu.read_yaml(yaml_fn)
        self.assertDictEqual(self.features_dict, actual)

    @logit()
    def test_current_directory(self):
        logger.debug(f'current working dir is really {self._fu.current_directory()}')
        my_mock_dir = r'\synthesys\testing'
        with mock.patch('FileUtil.getcwd', return_value=my_mock_dir):
            actual = self._fu.current_directory()
            self.assertEqual(actual, my_mock_dir)

    def test_read_text_file(self):
        filename = self._fu.qualified_path(self.path, self.text_fn)
        how_many_lines = randrange(10) + 2
        self.create_text_file(filename, how_many_lines)
        expected = self.generate_text_lines(how_many_lines)
        actual = [x.rstrip() for x in self._fu.read_text_file(filename)] # must remove newline chars
        self.assertListEqual(expected, actual)

    @logit()
    def test_read_text_file_err(self):
        # test an IO error
        filename = self._fu.qualified_path(self.path, self.text_fn)
        with mock.patch('FileUtil.open', create=True) as mocked_open:
            mocked_open.side_effect = IOError()
            self._fu.read_text_file(filename)

    @logit()
    def test_read_yaml(self):
        keys = ['firstname', 'lastname', 'zip']
        vals = ['Rajah', 'Chacko', 28269]
        self.create_yaml(keys, vals)

        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        d = self._fu.read_yaml(yamlFile=qualifiedPath)
        logger.debug(f'Contents of yaml: {d}')
        self.assertEqual(list(d.keys()), keys)
        self.assertEqual(vals[0], d[keys[0]])

    @logit()
    @mock.patch('FileUtil.safe_load')
    def test_read_yaml_err(self, mock_obj):
        yaml_fn = self._fu.qualified_path(self.path, self.yaml)
        self.create_text_file(yaml_fn)
        mock_obj.side_effect = YAMLError('mock error')
        actual = self._fu.read_yaml(yamlFile=yaml_fn)
        self.assertIsNone(actual)

    @logit()
    def test_qualified_path(self):
        fn = 'test.txt'
        shouldBe = self.path + sep + fn
        actual = self._fu.qualified_path(self.path, fn)
        logger.debug(f'Actual: {actual}. shouldBe: {shouldBe}.')
        self.assertEqual(actual, shouldBe)
        pathArray = ['c:', 'temp']

    @logit()
    def test_split_qualified_path(self):
        fn = 'test.txt'
        qpath = self._fu.qualified_path(self.path, fn)
        splitpath, splitfn = self._fu.split_qualified_path(qpath, makeArray=False)
        self.assertEqual(splitpath, self.path)
        self.assertEqual(splitfn, fn)
        pathArray, splitfn = self._fu.split_qualified_path(qpath, makeArray=True)
        self.assertEqual(pathArray, ['c:', 'temp'])
        self.assertEqual(splitfn, fn)
        path = r'C:\Users\Owners\Documents\Tickers.csv'
        pathArray, splitfn = self._fu.split_qualified_path(path, makeArray=True)
        self.assertEqual(pathArray, ['C:', 'Users', 'Owners', 'Documents'])
        self.assertEqual(splitfn, 'Tickers.csv')

    @logit()
    def test_split_file_name(self):
        expected_file = "file"
        expected_ext = ".ext"
        expected_fn = expected_file + expected_ext
        # First test with just file.ext
        actual_file, actual_ext = self._fu.split_file_name(expected_fn)
        self.assertEqual(actual_file, expected_file)
        self.assertEqual(actual_ext, expected_ext)
        # Another test with path/file.ext
        qpath = self._fu.qualified_path(self.path, expected_fn)
        actual_file, actual_ext = self._fu.split_file_name(qpath)
        self.assertEqual(actual_file, expected_file)
        self.assertEqual(actual_ext, expected_ext)

    @logit()
    def test_file_exists(self):
        self.create_csv()
        qualifiedPath = self._fu.qualified_path(self.path, self.fn)
        self.assertTrue(self._fu.file_exists(qualifiedPath))
        qualifiedPath = self._fu.qualified_path(self.path, 'noSuchFile.xxd')
        self.assertFalse(self._fu.file_exists(qualifiedPath))

    @logit()
    def test_ensure_dir(self):
        self._fu.ensure_dir(self.path)
        self.assertTrue(self._fu.dir_exists(self.path))

    @logit()
    def test_delete_file(self):
        self.create_csv()
        qualifiedPath = self._fu.qualified_path(self.path, self.fn)
        # delete_file should return True the first time
        self.assertTrue(self._fu.delete_file(qualifiedPath))
        # but return false the second time.
        self.assertFalse(self._fu.delete_file(qualifiedPath))

    @logit()
    @mock.patch('FileUtil.remove')
    def test_delete_file_err(self, mock_obj):
        self.create_csv()
        expected_log_message = 'delete_file mocktest'
        mock_obj.side_effect = OSError(expected_log_message)
        qualifiedPath = self._fu.qualified_path(self.path, self.fn)
        with self.assertLogs(FileUtil.__name__, level='DEBUG') as cm:
            ans = self._fu.delete_file(qualifiedPath)
            self.assertFalse(ans)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_copy_file(self):
        self.create_csv()
        tmp_path = self._fu.qualified_path(self.path, 'tmp')
        qualifiedPath = self._fu.qualified_path(self.path, self.fn)
        self._fu.ensure_dir(tmp_path)
        self._fu.copy_file(qualifiedPath, tmp_path)
        self.assertTrue(self._fu.file_exists(qualifiedPath))
        self._fu.rmdir_and_files(tmp_path)

    @logit()
    @mock.patch('FileUtil.copy2')
    def test_copy_file_err(self, mock_obj):
        tmp_path = self._fu.qualified_path(self.path, 'tmp')
        qualifiedPath = self._fu.qualified_path(self.path, self.fn)
        expected_log_message = 'copy_file mocktest'
        mock_obj.side_effect = IOError(expected_log_message)
        with self.assertLogs(FileUtil.__name__, level='DEBUG') as cm:
            _ = self._fu.copy_file(qualifiedPath, tmp_path)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_getList(self):
        dir_name = r'c:\temp'
        flist = self._fu.getList(dir_name)
        logger.debug(f'All list is: {flist}')


    def isFile_side_effect(*args, **kwargs) -> bool:
        """
        Side effect for mocking test_get_files.
        Returns True if there is a .txt in the filename. Not great, but ok for mocking.
        :param args:
        :param kwargs:
        :return:
        """
        return mock_is_file(args[1])
    def isDir_side_effect(*args) -> bool:
        return mock_is_dir(args[1])

    @logit()
    @mock.patch('FileUtil.isfile')
    @mock.patch('FileUtil.listdir')
    def test_get_files(self, mock_listdir, mock_isfile):
        dir_name = r'\nosuchdir'
        file_list = ['filea.txt', 'fileb.txt', 'filec.txt', 'somedir']
        mock_listdir.return_value = file_list
        mock_isfile.side_effect = self.isFile_side_effect
        actual = self._fu.get_files(dir_name)
        expected = [f for f in file_list if mock_is_file(f)] # Condition must match isFile_side_effect
        self.assertListEqual(expected, actual)


    @logit()
    @mock.patch('FileUtil.isdir')
    @mock.patch('FileUtil.listdir')
    def test_get_dirs(self, mock_listdir, mock_isdir):
        dir_name = r'\nosuchdir'
        file_list = ['filea.txt', 'fileb.txt', 'filec.txt', 'somedir']
        mock_listdir.return_value = file_list
        mock_isdir.side_effect = self.isDir_side_effect
        actual = self._fu.get_dirs(dir_name)
        expected = [f for f in file_list if mock_is_dir(f)] # Condition must match isDir_side_effect
        self.assertListEqual(expected, actual)

    @logit()
    def test_getRecursiveList(self):
        dir_name = r'\nosuchdir'
        file_list = ['filea.txt', 'fileb.txt', 'filec.txt']
        actual = self._fu.getRecursiveList(dir_name)
        self.assertListEqual(actual, [])  # Since no such dir, should be empty list
        eu = ExecUtil()
        exec_file = eu.exec_file_path()
        dir_name, _ = self._fu.split_qualified_path(exec_file)
        logger.debug(f'dir name is: {dir_name}')

        with mock.patch('FileUtil.listdir', return_value=file_list):
            actual = self._fu.getRecursiveList(dir_name)
            expected = [self._fu.fully_qualified_path(dirPath=dir_name, filename=f) for f in file_list]
            self.assertListEqual(expected, actual)

    @logit()
    def test_load_logs_and_subdir_names(self):
        dir_name = r'\nosuchdir'
        file_list = ['filea.txt', 'fileb.csv', 'otherfile.txt']
        actual = self._fu.load_logs_and_subdir_names(dir_name)
        self.assertListEqual(actual, [])  # Since no such dir, should be empty list
        eu = ExecUtil()
        exec_file = eu.exec_file_path()
        dir_name, _ = self._fu.split_qualified_path(exec_file)
        logger.debug(f'dir name is: {dir_name}')

        with mock.patch('FileUtil.listdir', return_value=file_list):
            # Test with neither prefix nor suffix
            actual = self._fu.load_logs_and_subdir_names(dir_name)
            expected = [self._fu.fully_qualified_path(dirPath=dir_name, filename=f) for f in file_list]
            self.assertListEqual(expected, actual)
            # Test for suffixes ending in .txt
            suffix = '.txt'
            actual = self._fu.load_logs_and_subdir_names(dir_name, requiredSuffix=suffix)
            txt_only = [self._fu.fully_qualified_path(dirPath=dir_name, filename=f) for f in file_list if f.endswith(suffix)]
            self.assertListEqual(txt_only, actual)
            # Test for prefixes starting with 'file'
            prefix = 'file'
            actual = self._fu.load_logs_and_subdir_names(dir_name, requiredPrefix=prefix)
            file_only = [self._fu.fully_qualified_path(dirPath=dir_name, filename=f) for f in file_list if f.startswith(prefix)]
            self.assertListEqual(file_only, actual)

    @logit()
    @mock.patch('FileUtil.isfile')
    @mock.patch('FileUtil.listdir')
    def test_cull_existing_files(self, mock_listdir, mock_isfile):
        dir_name = r'\nosuchdir'
        file_list = ['filea.txt', 'fileb.txt', 'filec.txt', 'somedir']
        mock_listdir.return_value = file_list
        mock_isfile.side_effect = self.isFile_side_effect
        qualified_file_list = [self._fu.qualified_path(dirPath=dir_name, filename=f) for f in file_list]
        actual = self._fu.cull_existing_files(qualified_file_list)
        expected = [f for f in qualified_file_list if mock_is_file(f)] # Condition must match isFile_side_effect
        self.assertListEqual(expected, actual)

    @logit()
    def test_read_generator(self):
        filename = self._fu.qualified_path(self.path, self.text_fn)
        how_many_lines = 5
        self.create_text_file(filename, how_many_lines)
        lines_read_in = 0
        for i, line in enumerate(self._fu.read_generator(filename)):
            logger.debug(f'Read in line {i}, which contains <{line}>.')
            lines_read_in += 1
        self.assertEqual(how_many_lines, lines_read_in)


    @logit()
    @mock.patch('FileUtil.open')
    def test_read_generator_err(self, mock_open):
        expected_log_message = 'mocked error'
        mock_open.side_effect = IOError(expected_log_message)
        filename = self._fu.qualified_path(self.path, self.text_fn)
        with self.assertLogs(FileUtil.__name__, level='DEBUG') as cm:
            for i, line in enumerate(self._fu.read_generator(filename)):
                x = line # TODO: Redirect stream and capture IOError message
                logger.debug(f'Read in line {i}, which contains <{x}>.')
                self.assertIsNone(x)
            logger.debug(f'Caught exception message: {cm.output}')
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))

    @logit()
    def test_file_modify_time(self):
        start_time = self._du.as_timestamp()
        keys = ['greeting', 'farewell', ]
        vals = ['Hello', 'Goodbye', ]
        self.create_yaml(keys, vals)
        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        mod_time = self._fu.file_modify_time(qualifiedPath)
        mod_timestamp = self._du.as_timestamp(dt=mod_time)
        logger.debug(f'mod_time is {mod_timestamp}. start_time is {start_time}.')
        self.assertTrue((start_time - mod_timestamp) < .1 ) # asserting a difference of < 0.1 seconds.

    @logit()
    def test_file_modify_time2(self):
        start_time = self._du.as_timestamp()
        keys = ['greeting', 'farewell', ]
        vals = ['Hello', 'Goodbye', ]
        self.create_yaml(keys, vals)
        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        mod_time = self._fu.file_modify_time2(qualifiedPath)
        mod_timestamp = self._du.as_timestamp(dt=mod_time)
        self.assertTrue((start_time - mod_timestamp) < .1 ) # asserting a difference of < 0.1 seconds.

    @logit()
    def test_list_modules(self):
        mods = []
        for mod_name in self._fu.list_module_contents(module_name='itertools'):
            mods.append(mod_name)

        self.assertTrue('__docs__' in mods)

    @logit()
    def test_list_modules(self):
        doc = self._fu.list_module_attributes('itertools', True)
        logger.debug('{}'.format(doc))
        mods = []
        for mod_name in self._fu.list_modules(module_name='itertools'):
            mods.append(mod_name)

        self.assertTrue('__doc__' in mods)
        self.assertTrue('__name__' in mods)


if __name__ == "__main__":
    main(argv=['first-arg-is-ignored'], exit=False)
    logger.info('Done.')