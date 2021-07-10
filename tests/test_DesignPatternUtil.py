import logging
from unittest import TestCase
from time import sleep
from LogitUtil import logit
from DesignPatternUtil import EckertSingleton, LockingSingleton, SingletonMeta, TS_SingletonMeta

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
LongInit uses sleep to take 2 seconds to initialize. It's good for testing thread-safe objects.
"""
class Simple(object):
    def __init__(self):
        logger.debug('Creating Simple object')
        self._hidden_value = None

    @property
    def my_value(self):
        return self._hidden_value

    @my_value.setter
    def my_value(self, v):
        self._hidden_value = v

class LongInit(object):
    def __init__(self):
        _delay = 2
        logger.debug('start init of LongInit')
        self._hidden_value = None
        sleep(_delay)
        logger.debug('finished init of LongInit after {_delay} seconds')

    @property
    def my_value(self):
        return self._hidden_value

    @my_value.setter
    def my_value(self, v):
        self._hidden_value = v

class Test_EckertSingleton(TestCase):
    @logit()
    def test_eckert_singleton(self):
        # Test 1, single object
        first = EckertSingleton(Simple) # Bare class name passed in. Init not used.
        first_value = 42
        first.my_value = first_value
        self.assertEqual(first_value, first.my_value, "Fail test 1")

        # Test 2, new object should have the first value.
        second = EckertSingleton(Simple)
        self.assertEqual(first_value, second.my_value, "Fail test 2")

        # Test 3, both objects are actually the same singleton.
        new_value = 21
        first.my_value = new_value
        self.assertEqual(first, second, "Fail test 3")

class Test_LockingSingleton(TestCase):
    @logit()
    def test_locking_singleton_long_init(self):
        # Test 1, single object
        first = LockingSingleton(LongInit) # Note! Creating the class but does not call __init__.
        first_value = 42
        first.my_value = first_value
        self.assertEqual(first_value, first.my_value, "Fail test 1")

        # Test 2, new object should have the first value.
        second = LockingSingleton(LongInit)
        self.assertEqual(first_value, second.my_value, "Fail test 2")

        # Test 3, both objects are actually the same singleton.
        new_value = 21
        first.my_value = new_value
        self.assertEqual(first, second, "Fail test 3")
