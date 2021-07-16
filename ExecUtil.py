import logging
import platform
from inspect import getfile, currentframe
from os.path import abspath, join, dirname
from pathlib import Path, PurePath
from subprocess import Popen, PIPE
from sys import path

from FileUtil import FileUtil
from LogitUtil import logit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Author: Rajah Chacko
Class ExecUtil created 28Jun19.

"""

class ExecUtil:
    def executing_file(self):
        """
        This could throw a NameError if it's being called from anaconda.
        :return:
        """
        logger.debug('exec_file_path is about to attempt accessing __file__.')
        return globals()['__file__']

    def exec_file_path(self) -> str:
        """
        Get the full dir and file path we're executing from using __file__, or if not available (probably because you're using a Jupyter notebook)
        use os.path.abspath.
        """
        try:
            ex_file = self.executing_file()
            return ex_file
        except NameError as err:
            logger.debug('Exception message: {msg}'.format(msg=err))
            return abspath('')

    def executing_directory(self) -> str:
        """
        Get the current executing directory using executing_file and stripping off the filename.
        Note differences between Windows and Linux.

        :return:
        """
        fu = FileUtil()
        path, _ = fu.split_qualified_path(self.executing_file())
        logger.debug(f'executing file is {self.executing_file()}')
        logger.debug(f'path (minus filename) is {path}')
        return path

    def add_path(self, newPath:str) -> str:
        """
        Adds newPath to the existing sys path, if needed.
        :param newPath: gets added temporarily to the PYTHONPATH.
        :return: updated string
        """
        strPath = str(newPath) if isinstance(newPath, PurePath) else newPath

        if strPath in path:
            logger.warning(f'path: {newPath} is already on sys.path. (No action taken.)')
            return path
        else:
            logger.debug(f'Adding new path: {strPath} to sys.path.')
            path.append(strPath)
        return path

    def parent_folder(self):
        """
        Return the parent path of the current path.
=        :return: absolute path of parent
        """
        path = Path(self.executing_directory())
        return path.parent

    def add_executing_file(self):
        """
        This gets the current executing directory and adds it to the path, if needed.
        :return: updated PYTHONPATH
        """
        ex_path = self.executing_file()
        return self.add_path(ex_path)

    def exec_os_cmd(self, cmd:str) -> list:
        """
        Execute an OS command and return the lines of output as a list of strings.

        :param cmd: an OS string like 'ls' for Linux or 'dir' for Windows.
        :return: list of strings
        """
        resp = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, close_fds=True).communicate()[0]
        lines = resp.decode().split('\n')
        return list(filter(None, lines))

    def parse_hadoop_ls(self, lines: list) -> list:
        """
        This parses a hadoop fs -du command and returns an array of file sizes.
        :param lines: list of strings, one per line.
        :return: list of file sizes
        """
        firstWords = [x.split(' ')[0] for x in lines]
        ans = [int(x) for x in firstWords]
        return ans

    def calc_total_bytes(self, hadoop_path:str):
        cmd = 'hadoop fs -du ' + hadoop_path
        lines = self.exec_os_cmd(cmd)
        return sum(self.parse_hadoop_ls(lines))

    @staticmethod
    def which_platform() -> str:
        """
        Answer which platform is running.
        :return: platform string like "Linux"
        """
        return platform.system()

    @staticmethod
    @logit(showArgs=False, showRetVal=True)
    def which_architecture() -> str:
        """
        Answer whether this python shell is running in 32-bit or 64-bit mode.
        Got this from https://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os

        :return: '32bit' or '64bit'
        """
        return platform.architecture()[0]
