from typing import Callable, Union, List
import unittest.mock as mock
import inspect
from os import walk
import logging
# from LogitUtil import LogStream
from FileUtil import FileUtil
from LogitUtil import logit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def method1():
    return 'hello world'

def method2(methodToRun:Callable[[],str]):
    result = methodToRun()
    return result


def method3(list_or_str: Union[List[str], str]) -> str:
    return f'Detecting list_or_str is: {list_or_str}'


@logit(logModuleName='scRatch')
def my_function():
    logger.debug('inside my_function')
    return 'welcome to my_function'

for root, dirs, files in walk(r"C:\Users\Owner\Music"):
    print (f'root: {root}')
    print (f'dirs: {dirs}')
    print (f'files: {files}')

def a_function():
    a = r"C:\Users\Owner\Music"
    b = []
    c = [1, 2, 3]
    return a, b, c

import re
import os
class Sample:

    def __init__(self, f,  RE = r'(?P<id>\d+)', STRICT_MATCHING = False):
        self.file = f
        self.basename = os.path.basename(os.path.splitext(self.file)[0])

        print (f'about to compile and match using regex {RE}')
        re_ = re.compile(RE)  #
        match = re_.fullmatch if STRICT_MATCHING else re_.match  #

        #self.__dict__.update(self.match(self.basename).groupdict())

    @classmethod
    def valid(cls, f):
        basename, ext = os.path.splitext(os.path.basename(f))
        return cls.match(basename) and ext.lower() in ('.jpg', '.jpeg', '.png')


class DetailedSample(Sample):
    def __init__(self, f):
        RE = r'(?P<id>\d+)_(?P<dir>[lr])_(?P<n>\d+)'
        STRICT_MATCHING = True
        super(DetailedSample, self).__init__(f, RE, STRICT_MATCHING)

from collections import Counter
car_ids = [32, 2, 3, 7, 3, 4, 32, 1]
c = Counter(car_ids)
count_pairs = c.most_common() # Gets all counts, from highest to lowest.
print (f'Most frequent: {count_pairs[0]}')
n = 3
print (f'Least frequent {n}: {count_pairs[:-n-1:-1]}')