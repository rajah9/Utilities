from collections import defaultdict, namedtuple, OrderedDict
from typing import Union, List, Tuple
from itertools import compress, repeat, chain
import numpy as np
from Util import Util
from copy import copy
from pandas import Series

Ints = List[int]
Bools = List[bool]
Strings = List[str]
Floats = List[float]

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

    def sort_by_inner_dict(self, d:dict, field: str, is_descending: bool = False) -> dict:
        """
        Sort a dict of dicts by a key from the FIRST-LEVEL inner dictionary.

        :param d: dictionary, like test_dict = {'Nikhil' : { 'roll' : 24, 'marks' : 17},
             'Akshat' : {'roll' : 54, 'marks' : 12},
             'Akash' : { 'roll' : 12, 'marks' : 15}}
        :param field: key name in the first-level inner dictionary
        :param is_descending: Set True to sort from largest to smallest.
        :return: Dictionary sorted by inner dict key specified by field.
        """
        ans = OrderedDict(sorted(d.items(), key=lambda x: x[1][field], reverse=is_descending))
        return ans

    def sort_by_nested_dict(self, d:dict, field: str, is_descending: bool = False) -> dict:
        """
        Sort a dict of dicts by a key from a NESTED dictionary. This can be deeper than the first level.
        :param d: nested dictionary, like d5 =
        {'node1': {'input_file': {'filename': 'xyz', 'worksheet': 'sheet1'},
                'output_file': {'output_file_handle': 'out1'}},
        'node2': {'input_file': {'filename': 'xyz', 'worksheet': 'sheet1'},
                'output_file': {'output_file_handle': 'out2'}},
        'node3': {'input_file': {'filename': 'xyz', 'worksheet': 'sheet1'},
                'output_file': {'output_file_handle': 'out1'}}}
        :param field: key name in an inner dictionary, like 'output_file_handle'
        :param is_descending: Set True to sort from largest to smallest.
        :return: Dictionary sorted by inner dict key specified by field.
        """
        NestingDict = namedtuple('NestingDict', ['key', 'nestedDict', 'sortByMe'])
        # Create a working list with key, dict, and the key to sort by.
        # Place them in a named tuple.
        # This is a little clearer with the list comprehension spelled out.
        unsorted_dicts = []
        for k, v in d.items():
            nd = NestingDict(key=k, nestedDict=v, sortByMe=self.recursive_dict_search(v, field))
            unsorted_dicts.append(nd)

        # Now sort the named tuples.
        sorted_list = sorted(unsorted_dicts, key=lambda x: x.sortByMe, reverse=is_descending)
        # Place the sorted named tuples in an order.
        # This is a little clearer with the dict comprehension spelled out.
        d = {}
        for l in sorted_list:
            d[l.key] = l.nestedDict
        return d

    def recursive_dict_search(self, d: dict, key: str, default=None):
        """Return a value corresponding to the specified key in the (possibly
        nested) dictionary d. If there is no item with that key, return
        default. From https://stackoverflow.com/a/43184871/509840.
        Uses a stack of iterators pattern; see https://garethrees.org/2016/09/28/pattern/.
        """
        stack = [iter(d.items())]
        while stack:
            for k, v in stack[-1]:
                if isinstance(v, dict):
                    stack.append(iter(v.items()))
                    break
                elif k == key:
                    return v
            else:
                stack.pop()
        return default

    def remove_key(self, d: dict, key: str, default=None):
        """
        Remove the key from the given dictionary.
        SIDE EFFECT: d is altered.
        If not found, return default.
        Return the value at the given key.
        Adopted from https://stackoverflow.com/a/11277439/509840.

        :param d: dict to examine
        :param key: key to be removed
        :param default: what gets return for a KeyError
        :return: the key just removed.
        """
        ans = d.pop(key, default)
        return ans

    def sorted_list(self, lst: list, is_descending: bool = False) -> list:
        """
        Sort the list. Note: This does a sort in place, so the l passed in is sorted as a side-effect.
        :param lst: list to sort
        :param is_descending: True if you want it in reverse order.
        :return:
        """
        lst.sort(reverse=is_descending)
        return lst

    def sorted_set(self, s: set, is_descending: bool = False) -> list:
        """
        Convert the set to a list and return the sorted list.
        :param s: Set to be sorted
        :param is_descending: True if you want it in reverse order.
        :return: a sorted list
        """
        return sorted(s, reverse=is_descending)

    def list_max_and_min(self, lst: Union[Series, list]) -> Tuple[float, float]:
        """
        Return the min and max of the list.
        This does a (shallow) copy so as not to sort the list as a side-effect.
        :param lst:
        :return: max and min of the list.
        """
        if isinstance(lst, list):
            return max(lst), min(lst)
        # Falls through if it's a Series.
        return lst.max(), lst.min()


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

    @staticmethod
    def index_in_list(lst: list, find_me: Union[str, int, float], throws_error_if_not_found: bool = True) -> int:
        """
        Return the index of find_me within list. Throw a ValueError if it's not found (or -1 if you set throws_error_not_found to False)

        :param lst:
        :param find_me:
        :param throws_error_if_not_found: if True and not found, throw a ValueError.
        :return: The index of find_me within lst (or -1 if throws_error_if_not_found is set to False)
        """
        try:
            return lst.index(find_me)
        except ValueError as e:
            if throws_error_if_not_found:
                raise e
            else:
                return -1

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
          Complex = CollectionUtil.named_tuple('Complex', 'real imaginary')
          c = Complex(3, 2.5) # creates 3 + 2.5i
          print ('Created ' + c.real + ' + ' + c.imaginary + "i")
        :param clz: name of the class of this namedtuple (typically capitalized)
        :param fields: either a list of strings, one per field, or the fields separated by spaces.
        :return:
        """
        return namedtuple(clz, fields)

    @staticmethod
    def remove_first_occurrence(orig_list: list, remove_me: Union[str, int]) -> list:
        """
        Remove the first occurrence of remove_me from orig_list.
        :param orig_list:
        :return:
        """
        ans = orig_list.copy()
        ans.remove(remove_me)
        return ans

    @staticmethod
    def remove_all_occurrences(orig_list: list, remove_me: Union[str, int]) -> list:
        """
        Remove all occurrences of remove_me from the original list.
        From https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
        :param orig_list:
        :param remove_me:
        :return:
        """
        ans = list(filter(lambda a: a != remove_me, orig_list))
        return ans

    @staticmethod
    def list_of_x_n_times(x: Union[str, int, float], n: int) -> list:
        """
        Return a list of x, repeated n times.
        :param x: scalar, like 3.14
        :param n: times to repeat, like 2
        :return: list like [3.14, 3.14]
        """
        ans = list(repeat(x, n))
        return ans

    @staticmethod
    def repeat_elements_n_times(lst: Union[Strings, Ints, Floats], n: int) -> list:
        """
        Take each element of the list and repeat it n times.
        Inspired by: https://stackoverflow.com/questions/24225072/repeating-elements-of-a-list-n-times
        :param x: list, like [10, 20, 30]
        :param n: repeat factor, like 2
        :return: each el in x repeated n times, like [10, 10, 20, 20, 30, 30]
        """
        return list(chain.from_iterable(repeat(x, n) for x in lst))

    @staticmethod
    def slice_list(my_list: list, start_index: int = 0, end_index: int = None, step: int = 1):
        """
        Return a sliced list, given the start_index, end_index, and step.
        :param my_list: original list to be sliced
        :param start_index: 0-based first index to use. Defaults to 0 (the first el)
        :param end_index: end of list index. Defaults to None (which means the end of the list).
        :param step: how many to skip. 2 means skip every other. Default of 1 means don't skip.
        :return: sliced array
        """
        end_idx = end_index or len(my_list)
        return my_list[start_index:end_idx:step]

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
