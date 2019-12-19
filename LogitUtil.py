"""
Created 3Apr19, Rajah Chacko
"""
import logging
from functools import wraps

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
#logging.getLogger('test').addHandler(logging.NullHandler())
#logger = logging.getLogger(__name__)

"""
Class logit created 9Apr19, Rajah Chacko
Last modified 1Jul19.
Usage: Before a def statement add something like:
    @logit()
    @logit(showArgs=True, showRetVal=False)
    @logit(showArgs=False, showRetVal=True)
    @logit(showArgs=True, showRetVal=True)
"""

class logit:
    def __init__(self, showArgs=False, showRetVal=False, logModuleName:str=None):  # Could call with @logit(showArgs=True, showRetVal=False)
        self.showArgs = showArgs
        self.showRetVal = showRetVal
        self.logger = logging.getLogger(__name__)
        if logModuleName:
            self.logger = logging.getLogger(logModuleName)
        else:
            self.logger.handlers = []
            self.logger.addHandler(logging.NullHandler())

    @property
    def logger(self):
        """
        How to use this logger property:
          lg = logit()
          lg.logger.debug('Starting.')
        :return:
        """
        return self._logger

    @logger.setter
    def logger(self, l):
        self._logger = l

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            self.logger.debug('Entering {}.'.format(f.__name__))
            if self.showArgs:
                self.print_args(*args, **kwargs)
            retval = f(*args, **kwargs)
            if self.showRetVal:
                ret_str = f'{retval}' if retval else 'None'
                self.logger.debug(f'>> Return value is {ret_str}.')
            self.logger.debug(f'Exiting {f.__name__}.')
            return retval

        return wrapper

    def print_args(self, *args, **kwargs):
        for i in range(len(args)):
            self.logger.debug(f'>> {i}. {args[i]}')
        for name, val in kwargs.items():
            self.logger.debug(f'>> {name} = {val}')


