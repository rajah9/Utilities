from constraint import Problem, AllDifferentConstraint, ExactSumConstraint, InSetConstraint
import sys
from collections import defaultdict

from Util import Util
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
