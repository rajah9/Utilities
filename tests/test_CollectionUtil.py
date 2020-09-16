import logging
from unittest import TestCase
from CollectionUtil import CollectionUtil, NumpyUtil
import numpy as np

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
