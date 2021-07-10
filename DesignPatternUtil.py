import typing
import logging
from threading import Lock, Thread

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Following is adopted from Bruce Eckert OnlyOne Singleton.
https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
"""

class EckertSingleton(object):
    _instance = None

    def __new__(cls, o_cls):
        if cls._instance is None:
            logger.debug(f'Creating the {o_cls} object')
            cls._instance = super(EckertSingleton, cls).__new__(o_cls)
            # Put any initialization here.
        return cls._instance

"""
Added locking to above.
"""

class LockingSingleton(object):
    _instance = None
    __singleton_lock = Lock()

    def __new__(cls, o_cls: typing.TypeVar):
        if cls._instance is None:
            logger.debug('obtaining lock')
            logger.debug(f'o_cls is {o_cls}')
            with cls.__singleton_lock:
                if not cls._instance:
                    logger.debug(f'Creating the {o_cls} object')
                    cls._instance = super(LockingSingleton, cls).__new__(o_cls)
            # Put any initialization here.
        return cls._instance

"""
Here is a Naïve singleton from Refactoring Guru.
Uses Metaclass.
To invoke it, declare:
  class Singleton(metaclass=SingletonMeta):
https://refactoring.guru/design-patterns/singleton/python/example
"""
class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
"""
Here's a thread-safe version.
Invoke it with this definition:
  class TS_Singleton(metaclass=TS_SingletonMeta):
"""
class TS_SingletonMeta(type):
    """
    This is a thread-safe implementation of TS_Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the TS_Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # TS_Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the TS_Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the TS_Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


