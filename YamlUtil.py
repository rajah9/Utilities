# -*- coding: utf-8 -*-
"""
Created on 30Oct19.
Author: Rajah Chacko

YamlUtility reads a yaml file into a dictionary or namedtuple.

Interesting Python features.
* YamlUtil is a subclass of FileUtil.
* has getter and setter properties
** asnamedtuple has a getter (but no setter)
* stores a (sort of private) class variable, a dictionary.
** provides two different ways to access it
** provides a list of dictionary keys.
"""
import logging
from collections import namedtuple
from FileUtil import FileUtil
from typing import Union, List

Dictionaries = List[dict]

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class YamlUtil(FileUtil):
    _yaml_dict = {}
    def __init__(self, yaml_file:str):
        if self.file_exists(yaml_file):
            d = self.read_yaml(yaml_file)
            self.asdict = d
        else:
            logger.warning(f'Unable to find file {yaml_file}. Leaving dictionary empty.')

    @property
    def asdict(self):
        return self._yaml_dict
    @asdict.setter
    def asdict(self, d: Union[dict, Dictionaries]):
        if isinstance(d, dict):
            ans = d
        else:
            ans = {}
            for a_dict in d:
                for k, v in a_dict.items():
                    ans[k] = v

        self._yaml_dict = ans

    @property
    def asnamedtuple(self):
        YamlHelper = namedtuple('yaml_helper', self.fields)
        ans_tuple = YamlHelper(**self._yaml_dict)
        return ans_tuple

    @property
    def fields(self):
        return list(self._yaml_dict.keys())
