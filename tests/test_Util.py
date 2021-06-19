import logging
from unittest import TestCase

from Util import Util

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestUtil(TestCase):
    def test_instantiate(self):
        expected_log_message = 'Starting'
        with self.assertLogs(Util.__name__, level='INFO') as cm:
            u = Util()
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
