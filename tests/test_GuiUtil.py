import logging
import sys
from unittest import TestCase

from GuiUtil import GuiUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Test_GuiUtil(TestCase):
    def setUp(self):
        # self.gu = GuiUtil()
        pass

    def test_set_clipboard(self):
        expected = "xyzzy"
        GuiUtil.set_clipboard(expected)
        actual = GuiUtil.get_clipboard()
        logger.debug(f'Contents of clipboard: {actual}')
        self.assertEqual(expected, actual)

