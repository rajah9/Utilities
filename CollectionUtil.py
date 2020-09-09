from collections import defaultdict
from typing import Union, List

import numpy as np

from Util import Util

Ints = List[int]

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
