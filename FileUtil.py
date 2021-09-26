# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 13:32:50 2018
@author: Rajah Chacko
"""

import copy
import importlib
import logging
import platform
import posixpath
from datetime import datetime, time
from os import remove, makedirs, sep, stat, listdir, error
from os.path import split, join, normpath, isdir, getctime, getmtime
from pathlib import Path
from shutil import copy2, rmtree
from typing import Dict, Tuple, List, Union

from yaml import safe_load, YAMLError, dump

from DateUtil import DateUtil
from LogitUtil import logit

Strings = List[str]

_IS_PROD = True

"""
Class FileUtil last updated 9Sep19.
This class does basic file utilities such as reading and writing text files and
moving files from one directory to another. 

To import this library, here are handy import statements and instantiation:
from Utilities.FileUtil import FileUtil
fu = FileUtil()

Interesting Python facets:
* Uses Tuple to return two values
* Uses _IS_PROD to create either a regular logger or a null logger.
"""


class FileUtil:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if _IS_PROD:
            self.logger.handlers = []
            self.logger.addHandler(logging.NullHandler())
        self._isWindows = platform.system() == 'Windows'

    @property
    def is_Windows(self) -> bool:
        return self._isWindows

    @property
    def separator(self) -> str:
        # This could return os.sep, but because I may be testing Linux on a Windows env, I'm making it a property.
        if self.is_Windows:
            return '\\'
        return '/'

    @staticmethod
    def current_directory() -> str:
        """
        Get the current working directory.
        :return:
        """
        return Path().resolve()

    def read_text_file(self, filename: str, encoding='utf8') -> Strings:
        """
        Read the given file and return its lines as a list, one per line.
        """
        try:
            with open(filename, 'r', encoding=encoding) as f:
                return f.readlines()
        except (UnicodeDecodeError, IOError) as err:
            self.logger.error('Exception type: {typ}'.format(typ=type))
            self.logger.error('Exception message: {msg}'.format(msg=err))
            return []

    @logit()
    def read_generator(self, filename: str) -> str:
        """
        Read the given file and return its lines as a list, one line for each call.
        Use this to minimize the memory required for large text files.

        :param filename: the text file to read from.
        :return:  one line (or None in case of an exception).
        """
        self.logger.debug('entering read_generator')
        try:
            with open(filename, 'r') as f:
                for line in f:
                    yield line
        except IOError as err:
            self.logger.error('Exception message: {msg}'.format(msg=err))
            yield None

    def write_text_file(self, filename: str, lines: list, encoding: str = 'utf8') -> None:
        """
        Write the given lines to the given filename, one per line.
        """
        with open(filename, "w", encoding=encoding) as f:
            for line in lines:
                f.write(f'{line}\n')

    def get_list(self, dir_path: Union[str, Path]) -> list:
        """
        Return a list of files and directories in the given dirPath. This is not recursive.
        :param dir_path:
        :return: list of files and dirs.
        """
        p = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        all = [f for f in p.iterdir()]
        ans = [x.name for x in all]
        return ans
    """
    TODO: working area 
    """

    def generate_path(self, dir_path: Union[str, Path] = '.', filename: str = None) -> Path:
        """
        Ensure a dir exists for the given dir_path.
        It does NOT create the file if one is given.
        :param dir_path: Path to file or dir. If omitted, use the current working dir.
        :param filename: (optional)
        :return: Path to dir_path.
        """
        dir_path = Path(dir_path).resolve()
        path = dir_path / filename if filename else dir_path
        if dir_path.exists():
            self.logger.debug(f'Path (or file) {dir_path} exists.')
            return dir_path
        else:
            if path.is_file():
                parent = path.parent
                self.logger.debug(f'Creating parent dir: {parent}')
                parent.mkdir(exist_ok=True, parents=True)
                return parent
        self.logger.debug(f'Creating path {dir_path}')
        dir_path.mkdir(exist_ok=True, parents=True)
        return dir_path

    def get_files_and_dirs(self,  dir_path: Union[str, Path]) -> list:
        """
        Given a directory path, return a list of files and directories.
        """
        p = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        return [f for f in p.iterdir()]

    def file_or_dir_exists(self, dir_or_file_path: Path) -> bool:
        """
        Return True if this dir or file exists.
        :param dir_or_file_path:
        :return:
        """
        return dir_or_file_path.exists()

    def get_files(self, dir_path: Union[str, Path]) -> list:
        """
        Return a list of files in a directory.
        """
        p = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        all = [f for f in p.iterdir() if f.is_file()]
        ans = [f.name for f in all]
        return ans

    def get_dirs(self, dir_path: str) -> list:
        """
        Return a list of subdirs in a directory.
        """
        p = dir_path if isinstance(dir_path, Path) else Path(dir_path)
        all = [d for d in p.iterdir() if d.is_dir()]
        ans = [d.name for d in all]
        return ans

    def get_recursive_list(self, dir_path: str) -> list:
        """
        For the given path, get the list of all files in the directory tree.
        The list is a qualified path, for example,
          ['\\C:\\Users\\Owner\\PycharmProjects\\Utilities\\filea.txt', '\\C:\\Users\\Owner\\PycharmProjects\\Utilities\\fileb.txt' ]
        """
        # Create a list of file and subdirectory names in the given directory
        if not self.dir_exists(dir_path):
            return []

        list_of_files = self.get_list(dir_path)
        all_files = list()

        for entry in list_of_files:
            full_path = self.fully_qualified_path(dir_path, entry)
            if self.dir_exists(full_path):
                all_files = all_files + self.get_recursive_list(full_path)
            else:
                all_files.append(full_path)
        return all_files

    @logit(showArgs=True, showRetVal=False)
    def load_logs_and_subdir_names(self, root_path=str, required_prefix=None, required_suffix=None) -> list:
        """
        Given the rootPath, traverse the subdirectories and return a list with the
        given prefix (if any) and suffix (if any).

        The list is a qualified path, for example,
          ['\\C:\\Users\\Owner\\PycharmProjects\\Utilities\\filea.txt', '\\C:\\Users\\Owner\\PycharmProjects\\Utilities\\fileb.txt' ]
        """
        qualified_files = self.get_recursive_list(root_path)
        ans = []

        for full_path in qualified_files:
            _, fn = self.split_qualified_path(full_path)
            include = True
            if required_prefix and not fn.startswith(required_prefix):
                include = False
            if required_suffix and not fn.endswith(required_suffix):
                include = False
            if include:
                ans.append(full_path)
        return sorted(ans)  # This sorts logfiles chronologically

    def cull_existing_files(self, file_list: list) -> list:
        """
        Cull the given fileList (containing full paths) and return only the existing files.
        This is useful if some of the files in the list have been deleted.
        """
        return [f for f in file_list if self.file_exists(f)]

    def qualified_path_old(self, dir_path: str, filename: str, dir_path_is_array: bool = False) -> str:
        """
        From the given dir and filename, return the qualified path.
        11Dec18 Fixed a problem that was making the dirPath longer by creating a deepcopy.
        28Apr21 Fixed a Windows vs Linux variation.
        """
        def combined_file_components(file_component_array: list) -> str:
            """
            Given an array of file components, provide a platform-appropriate absolute path.
            :param file_component_array: list like ['c:', 'temp','subdir','subsubdir']
            :return: string like 'c:\temp\subdir\subsubdir'
            """
            self.logger.debug(f'Before splat: {file_component_array}')
            # If it's Windows, add an extra separator.
            if self.is_Windows:
                file_component_array.insert(1, self.separator)
            return join(*file_component_array) # The * (splat) helps os.path.join treat the args as one arg.

        if dir_path_is_array:
            r = copy.deepcopy(dir_path)
            r.append(filename)
            return combined_file_components(r)
        else:
            # dirPath is a string; simply join it.
            # return join(dirPath, filename) Doesn't work if testing Linux in a Windows env!
            return dir_path + self.separator + filename

    def qualified_path(self, dir_path: str, filename: str, dir_path_is_array: bool = False) -> Path:
        """
        From the given dir and filename, return the qualified path.
        Refactored qualified_path to use pathlib.
        :param dir_path:
        :param filename:
        :param dir_path_is_array:
        :return:
        """
        p = Path(*dir_path) if dir_path_is_array else Path(dir_path)
        return p.joinpath(filename)

    def fully_qualified_path(self, dir_path: str, filename: str, dir_path_is_array: bool = False) -> str:
        """
        From the given dir and filename, return a qualified path. Prepend a / for Linux systems.
        """
        ans = self.qualified_path(dir_path, filename, dir_path_is_array)
        if not self.is_Windows:
            parts = ans.parts
            if parts[0] != ans.root:
                q = '/' / ans
                return q
        return ans

    @logit(showArgs=True, showRetVal=True)
    def split_qualified_path(self, qualified_path: str, make_array=False):
        """
        From the given qualified path, return the dir and filename.
        if makeArray is False, make the path a string, like '/users/owner/'
        if makeArray is True, make the path an array, like ['users', 'owner']
        return a tuple of (path string, filename).
        """
        path, filename = split(qualified_path)
        if make_array:
            path = normpath(path)
            path_array = path.split(sep)
            return path_array, filename
        else:
            path, filename = split(qualified_path)
            return path, filename

    @logit()
    def split_file_name(self, path_or_fn: str) -> Tuple[str, str]:
        """
        Takes a full path name like /home/a137078/file.ext or a filename like file.ext and returns file and .ext.
        :param path_or_fn:
        :return: f filename and ext extension (with a .)
        """
        _, fn = self.split_qualified_path(path_or_fn)
        f = posixpath.splitext(fn)[0]
        ext = posixpath.splitext(fn)[1]
        return f, ext

    @staticmethod
    def file_exists(qualifiedPath: Union[str, Path]) -> bool:
        """
        Return true if the file exists.
        """
        test_me = qualifiedPath if isinstance(qualifiedPath, Path) else Path(qualifiedPath)
        return test_me.exists()

    @staticmethod
    def is_file(qualifiedPath: Union[str, Path]) -> bool:
        """
        Return true if this is a file (as opposed to a directory).
        """
        test_me = qualifiedPath if isinstance(qualifiedPath, Path) else Path(qualifiedPath)
        return test_me.is_file()

    def delete_file(self, qualified_path: str) -> bool:
        """
        Delete the given file. Return true if successful.
        """
        self.logger.debug(f'Attempting to delete {qualified_path}.')
        if not self.file_exists(qualified_path):
            self.logger.error(f'File {qualified_path} does not exist, so it cannot be removed.')
            return False

        try:
            remove(qualified_path)
        except OSError as err:
            self.logger.error('Exception message on {qualifiedPath}: {msg}'.format(qualifiedPath=qualified_path, msg=err))
            return False

        return True

    # @logit(showArgs=True, showRetVal=False)
    def ensure_dir(self, dir_path):
        """
        Check if the directory exists, and if it does not, create it.
        """
        if not self.dir_exists(dir_path):
            self.logger.debug(f'Creating directory {dir_path}.')
            makedirs(dir_path)
        else:
            self.logger.debug(f'The directory {dir_path} exists.')

    def dir_exists(self, dir_path) -> bool:
        """
        Return True if and only if the directory exists.
        """
        return isdir(dir_path)

    def rmdir_and_files(self, dir_path: str):
        """
        Delete any files within the dir_path and remove the dir itself.
        """
        self.logger.debug(f'Deleting the files and removing directory {dir_path}.')
        rmtree(dir_path, ignore_errors=True)

    def copy_file(self, source_file: str, destination: str) -> bool:
        """
        Copy the source file to the destination. destination may be a qualified file (with a complete directory and a
        file name) or a simple directory.
        Return True if the copy was successful.
        """
        try:
            copy2(source_file, destination)
            self.logger.debug(f'Copied file {source_file} to {destination}.')
            return True
        except FileNotFoundError as err:
            self.logger.error('Exception message: {msg}'.format(msg=err))
            self.logger.error(f'File {source_file} not found.')
            return False
        except (IOError, error) as err:
            self.logger.error('Exception message: {msg}'.format(msg=err))
            return False

    def read_yaml(self, yamlFile: str = './example.yaml') -> Dict:
        with open(yamlFile, 'r') as f:
            try:
                return safe_load(f)
            except YAMLError as err:
                self.logger.error('YAML error: {msg}'.format(msg=err))
                return None

    def dump_yaml(self, yamlFile: str = './example.yaml', d: Dict = {'line1': 'A', 'line2': 2}, use_default_flow_style: bool = False):
        with open(yamlFile, 'w') as f:
            dump(d, f, default_flow_style=use_default_flow_style)

    @logit(showArgs=True, showRetVal=True)
    def file_modify_time(self, filename: str) -> datetime:
        """
        ctime returns a formatted datetime like this: Thu Apr 11 10:36:18 2019

        :param filename:
        :return: the datetime (timezone unaware!)
        """
        du = DateUtil()
        if self.is_Windows:
            date_str = getctime(filename)
            ans = du.asDate(date_str, myFormat=DateUtil.iso_format)
        else:
            date_str = time.ctime(getmtime(filename))
            ans  = du.asDate(date_str, myFormat=DateUtil.ctime_format)
        return ans

    @logit(showArgs=True, showRetVal=True)
    def file_modify_time2(self, filename: str) -> float:
        """
        mtime returns a timestamp (float).
        :param filename:
        :return: the datetime (timezone unaware!)
        """
        statinfo = stat(filename)
        return statinfo.st_mtime

    @logit()
    def file_size(self, filename: str) -> int:
        """
        Return the file size.
        :param filename:
        :return:
        """
        statinfo = stat(filename)
        return statinfo.st_size


    @logit()
    def list_modules(self, module_name: str):
        mod = importlib.import_module(module_name)
        for mod_entry in dir(mod):
            yield mod_entry

    @logit()
    def list_module_attributes(self, module_name: str, doc_only: bool = False):
        mod = importlib.import_module(module_name)
        if doc_only:
            yield mod['__doc__']
        else:
            for attr, val in vars(mod).items():
                yield attr, val
