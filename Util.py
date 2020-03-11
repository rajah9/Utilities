import logging
from LogitUtil import logit

class Util:
    def __init__(self, null_logger:bool=False):
        if not null_logger:
            self.logger = self.init_logger()
            self.logger.info('Starting')
        else:
            print ('No logging.')
            self.logger = logging.getLogger(__name__)
            self.logger.handlers = []
            self.logger.addHandler(logging.NullHandler())


    @logit()
    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)
        return self.logger