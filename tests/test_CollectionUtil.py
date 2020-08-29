import logging
from unittest import TestCase
from CollectionUtil import CollectionUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Test_CollectionUtil(TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_CollectionUtil, self).__init__(*args, **kwargs)
        self._cu = CollectionUtil()

    def test_counter(self):
        test1 = "the rain in spain stays mainly in the plain"
        test1_list = test1.split()
        actual = self._cu.counter(test1_list)
        self.assertEqual(1, actual['spain'])
        self.assertEqual(2, actual['in'])

    def test_sort_by_value(self):
        unsorted_dict = {'three': 3, 'two': 2, 'ten': 10, 'one': 1}
        # Test 1, ascending
        sort_asc = self._cu.sort_by_value(unsorted_dict)
        actual1 = list(sort_asc.values())
        expected1 = sorted(unsorted_dict.values())
        self.assertListEqual(expected1, actual1)
        # Test 2, descending
        sort_dsc = self._cu.sort_by_value(unsorted_dict, is_descending=True)
        actual2 = list(sort_dsc.values())
        expected2 = sorted(unsorted_dict.values(), reverse=True)
        self.assertListEqual(expected2, actual2)
