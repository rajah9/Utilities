import logging
from unittest import TestCase
from CollectionUtil import CollectionUtil, NumpyUtil
import numpy as np
from copy import deepcopy
from pandas import Series

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

    def test_sorted_list(self):
        # Test 1, ascending
        unsorted_list = [5, 7, 3, 1, 10, 0]
        is_reversed = False
        exp1 = deepcopy(unsorted_list)
        exp1.sort(reverse=is_reversed)
        self.assertListEqual(exp1, self._cu.sorted_list(unsorted_list, is_descending=is_reversed), "Fail test 1")
        # Test 2, descending
        unsorted_list = [5, 7, 3, 1, 10, 0]
        is_reversed = True
        exp2 = deepcopy(unsorted_list)
        exp2.sort(reverse=is_reversed)
        self.assertListEqual(exp2, self._cu.sorted_list(unsorted_list, is_descending=is_reversed), "Fail test 2")

    def test_list_max_and_min(self):
        # Test 1, ascending
        unsorted_list = [5, 7, 3, 1, 10, 0]
        orig = deepcopy(unsorted_list)
        act_max, act_min = self._cu.list_max_and_min(unsorted_list)
        self.assertEqual(0, act_min, 'Test 1 min fail')
        self.assertEqual(10, act_max, 'Test 1 max fail')
        self.assertListEqual(orig, unsorted_list, 'Test 1 fail: original list was modified')
        # Test 2, using a series.
        unsorted_series = Series(unsorted_list)
        act_max, act_min = self._cu.list_max_and_min(unsorted_series)
        self.assertEqual(0, act_min, 'Test 2 min fail')
        self.assertEqual(10, act_max, 'Test 2 max fail')

    def test_layout(self):
        # Test 1, 2 x 3, by rows
        exp1 = np.array([[0, 1, 2],
                         [3, 4, 5]]).tolist()
        act1 = self._cu.layout(rows=2, cols=3, row_dominant=True).tolist()
        self.assertListEqual(exp1, act1, "test 1 fail")

        # Test 2, 2 x 3, by columns
        exp2 = np.array([[0, 2, 4],
                         [1, 3, 5]]).tolist()
        act2 = self._cu.layout(rows=2, cols=3, row_dominant=False).tolist()
        self.assertListEqual(exp2, act2)
        # Test 3, 1 x 3, by rows, non-sequential table order
        list3 = [3, 6, 42]
        exp3 = np.array([list3])
        act3 = self._cu.layout(rows=1, cols=3, row_dominant=True, tiling_order=list3)
        self.assertListEqual(exp3.tolist(), act3.tolist())

    def test_indices_of_True(self):
        # Test 1, small test
        exp1 = [0, 7, 10, 14]
        def mark_true(size: int, list_to_mark: list) -> list:
            # Init a bool list of the given size to False. Then set the indices in list_to_mark as True
            ans = [False] * size
            for i in list_to_mark:
                ans[i] = True
            return ans

        test1 = mark_true(15, exp1)
        act1 = self._cu.indices_of_True(test1)
        self.assertListEqual(exp1, act1)
        # Test 2, big(ger) test
        exp2 = [0, 7, 10, 11, 14, 20]
        test2 = mark_true(21, exp2)
        act2 = self._cu.indices_of_True(test2)
        self.assertListEqual(exp2, act2)

    def test_any_string_contains(self):
        # Test 1, yes, it's there
        test1 = ["To be, or not to be: that is the question:", "Whether 'tis nobler in the mind to suffer",
                 "The slings and arrows of outrageous fortune,", "Or to take arms against a sea of troubles,",
                 "And by opposing end them?"]
        self.assertTrue(self._cu.any_string_contains(test1, "slings"))
        # Test 2, yes, it's there in the first position
        self.assertTrue(self._cu.any_string_contains(test1, "And"))
        # Test 3, yes, it's there in the last position
        self.assertTrue(self._cu.any_string_contains(test1, "suffer"))
        # Test 4, no, it's not there
        self.assertFalse(self._cu.any_string_contains(test1, "this"))

    def test_dict_comprehension(self):
        # Test 1, basic
        keys = ['one', 'ten', 'hundred', 'thousand']
        values = [1, 10, 100, 1000]
        exp = {}
        for k, v in zip(keys, values):
            exp[k] = v
        act = self._cu.dict_comprehension(keys, values)
        self.assertEqual(exp, act)

    def test_replace_elements_in_list(self):
        # Test 1, basic
        test1 = ['a', 'b', 'x']
        exp1 =  ['a', 'b', 'c']
        act1 = self._cu.replace_elements_in_list(before_list=test1, find_me='x', replace_me='c')
        self.assertListEqual(exp1, act1)

    def test_named_tuple(self):
        # Test 1, using a list
        name = 'Bill'
        age = 35
        Person = CollectionUtil.named_tuple('Person', ['name', 'age'])
        william = Person(name, age)
        self.assertEqual(name, william.name)
        self.assertEqual(age, william.age)
        # Test 2, using a string for the fields
        Person2 = CollectionUtil.named_tuple('Person', 'name age')
        billy = Person2(name, age)
        self.assertEqual(name, billy.name)
        self.assertEqual(age, billy.age)

    def test_dict_comprehension(self):
        # Test 1, simple dict
        list2 = ['alfa', 'bravo', 'charlie', 'delta']
        list1 = [1, 2, 3, 4]
        exp_dict = {k:v for (k,v) in zip(list1, list2)}
        act_dict = self._cu.dict_comprehension(list1, list2)
        self.assertEqual(exp_dict, act_dict)

    def test_remove_first_occurrence(self):
        # Test 1.
        orig = [1, 2, 3, 2, 2, 2, 3, 4]
        x = orig.copy()
        remove_me = 3
        exp = x.copy()
        exp.remove(remove_me)
        act =  self._cu.remove_first_occurrence(x, remove_me)
        self.assertListEqual(exp, act)
        self.assertListEqual(orig, x) # Original list should be unchanged.

    def test_remove_all_occurrences(self):
        orig = [1, 2, 3, 2, 2, 2, 3, 4]
        x = orig.copy()
        remove_me = 3
        exp = list(filter((remove_me).__ne__, x))
        self.assertListEqual(exp, self._cu.remove_all_occurrences(x, remove_me))
        self.assertListEqual(orig, x) # Original list should be unchanged.

    def test_list_of_x_n_times(self):
        # Test 1, float
        scalar = 3.14
        times = 4
        exp = [scalar] * times
        self.assertListEqual(exp, CollectionUtil.list_of_x_n_times(scalar, times))
        # Test 2, int
        scalar = 42
        times = 12
        exp = [scalar] * times
        self.assertListEqual(exp, CollectionUtil.list_of_x_n_times(scalar, times))
        # Test 3, string
        scalar = 'Figaro'
        times = 3
        exp = [scalar] * times
        self.assertListEqual(exp, CollectionUtil.list_of_x_n_times(scalar, times))
        # Test 4, list
        my_list = ['do', 'be']
        times = 2
        exp = [my_list, my_list]
        self.assertListEqual(exp, CollectionUtil.list_of_x_n_times(my_list, times))

    def test_slice_list(self):
        # Test 1, whole list
        act = list(range(10,100,10)) # [10 .. 90]
        exp1 = list(range(10,100,10))
        self.assertListEqual(exp1, CollectionUtil.slice_list(act), 'failed test 1')
        # Test 2, start on second el and skip every other
        exp2 = [20, 40, 60, 80]
        self.assertListEqual(exp2, CollectionUtil.slice_list(act, start_index=1, step=2), 'failed test 2')
        # Test 3, end early
        end_here = 5
        exp3 = act[0:end_here]
        self.assertListEqual(exp3, CollectionUtil.slice_list(act, end_index=end_here), 'failed test 3')
        # Test 4, all params
        start_here = 1
        my_step = 3
        exp4 = act[start_here:end_here:my_step]
        self.assertListEqual(exp4, CollectionUtil.slice_list(act, start_index=start_here, end_index=end_here, step=my_step), 'failed test 4')
        # Test 5, step = 1
        exp5 = exp1
        self.assertListEqual(exp5, CollectionUtil.slice_list(act, step=1), 'failed test 5')



class Test_NumpyUtil(TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_NumpyUtil, self).__init__(*args, **kwargs)
        self._nu = NumpyUtil()

    def test_to_numpy_array(self):
        # Test 1, mixed ints and floats should go to floats
        list1 = [3, 4, 3.14159]
        exp1 = np.array([x * 1.0 for x in list1])
        act1 = self._nu.to_numpy_array(list1)
        self.assertListEqual(exp1.tolist(), act1.tolist())
        # Test 2, same as test 1, but explicitly set dtype to float
        exp2 = np.array([x * 1.0 for x in list1])
        act2 = self._nu.to_numpy_array(list1, dtype=np.float)
        self.assertListEqual(exp2.tolist(), act2.tolist())
        # Test 3, same as test 1, but explicitly set dtype to int
        exp3 = np.array([int(x) for x in list1])
        act3 = self._nu.to_numpy_array(list1, dtype=np.int)
        self.assertListEqual(exp3.tolist(), act3.tolist())
