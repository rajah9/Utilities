import logging
from unittest import mock, TestCase, main
from EmitterUtil import EmitterUtil
from LogitUtil import logit
from CollectionUtil import CollectionUtil
from datetime import datetime

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
"""

class Test_EmitterUtil(TestCase):
    def setUp(self):
        self._eu = EmitterUtil()

    @logit()
    def test_emit(self):
        self._eu.emit()

    @logit()
    def test_preamble(self):
        cu = CollectionUtil()
        currentYear = datetime.today().strftime("%Y")
        exp = f'Copyright {currentYear}'
        act = self._eu.preamble()
        self.assertTrue(cu.any_string_contains(lines=act, find_me=exp))


if __name__ == '__main__':
    main()
