from collections import defaultdict, namedtuple
from typing import Union, List
from itertools import compress
import numpy as np

from Util import Util

Ints = List[int]
Bools = List[bool]
Strings = List[str]

"""
Interesting Python Featuers:
* implements default dictionary to count words
"""

class CollectionUtil(Util):
    def __init__(self):
        super(CollectionUtil, self).__init__(null_logger=False)

    def counter(self, count_this_list: list) -> dict:
        """
        Count the objects (usually strings that have been lowercased) and return a dictionary
        :param count_me: list of primitive objects to count
        :return: dict like {word1: count1 ... }
        """
        ans = defaultdict(int)
        for item in count_this_list:
            ans[item] += 1
        return ans

    def sort_by_value(self, d: dict, is_descending: bool = False) -> dict:
        """
        Sort the dictionary by value.
        :param d: input dictionary
        :param is_descending: Set True to sort from largest to smallest.
        :return: sorted dictionary
        """
        return {k: v for k, v in sorted(d.items(), reverse=is_descending, key=lambda item: item[1])}

    def layout(self, rows: int, cols: int, row_dominant: bool = True, tiling_order: Ints = None) -> np.array:
        """
        Given rows and columns (and whether tile_by_rows is True or False), return a
        rows x cols array of ascending numbers (or the numbers in tables_to_tile). If rows = 2 and cols = 3 and tile_by_rows is True, return
          [[0 1 2],
          [3 4 5]]
        If rows = 2 and cols = 3 and tile_by_rows is False, return
          [[0 2 4],
          [1 3 5]]
        :param rows: number of rows
        :param cols: number of columns
        :param row_dominant: if True, the sequence goes l to r. If False, from top to bottom.
        :param tiling_order: if present, use instead of 0,1,...,rows*cols-1
        :return: np.array
        """
        if tiling_order:
            a = np.array(tiling_order)
        else:
            a = np.arange(0, rows*cols)
        if row_dominant:
            return a.reshape(rows, cols)
        return a.reshape(cols, rows).transpose()

    def indices_of_True(self, bool_list: Bools) -> np.array:
        """
        Given an array of bool, provide the indices of those that are True.
        From https://stackoverflow.com/a/21448251
        :param bool_list:
        :return:
        """
        if len(bool_list) < 15: # Need to do better measurements of threshold
            return list(compress(range(len(bool_list)), bool_list))
        return list(np.where(bool_list)[0])

    def any_string_contains(self, lines: Strings, find_me: str) -> bool:
        """
        Return True iff any of the strings contains find_me.
        :param lines: List of strings in which to search
        :param find_me: str to search for
        :return: True iff any of the strings contains find_me.
        """
        return any(line.find(find_me) >= 0 for line in lines)

    def dict_comprehension(self, keys: list, values: list) -> dict:
        """
        Create a dictionary comprehension of the key-value pairs
        :param keys: list
        :param values: list
        :return: dict
        """
        return { k:v for (k,v) in zip(keys, values)}

    def replace_elements_in_list(self, before_list: list, find_me: Union[str, int, float], replace_me: Union[str, int, float]) -> list:
        """
        Find the elements in before_list and replace whole elements as specified.
        :param before_list: list to change
        :param find_me: element to find within the list
        :param replace_me: element to replace with
        :return: new list with replaced elements
        """
        return [replace_me if el == find_me else el for el in before_list]


    @staticmethod
    def named_tuple(clz: str, fields: Union[str, Strings]) :
        """
        Create a namedtuple.
        Example use:
          Complex = CollectionUtil('Complex', 'real imaginary')
          c = Complex(3, 2.5) # creates 3 + 2.5i
          print ('Created ' + c.real + ' + ' + c.imaginary + "i")
        :param clz: name of the class of this namedtuple (typically capitalized)
        :param fields: either a list of strings, one per field, or the fields separated by spaces.
        :return:
        """
        return namedtuple(clz, fields)

class NumpyUtil(Util):
    def __init__(self):
        super(NumpyUtil, self).__init__(null_logger=False)

    def to_numpy_array(self, data: list, dtype: Union[str, list] = None) -> np.array:
        """
        Return a numpy array.
        See https://numpy.org/doc/stable/user/basics.types.html for what you can place in dtype.
        If dtype is None, numpy will make a best guess.
        :param data: usually an iterable, such as a list.
        :param dtype: None, or a dtype such as np.uint, np.float.
        :return: a np.array
        """
        if dtype:
            return np.array(data, dtype=dtype)
        return np.array(data)
