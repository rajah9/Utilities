import sys
from pathlib import Path
import logging
from unittest import mock, TestCase, main
from GuiUtil import GuiUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Test_GuiUtil(TestCase):
    def setUp(self):
        # self.gu = GuiUtil()
        pass

    def test_which_platform(self):
        plat = "Windows" if "win" in sys.platform else "Linux"
        self.assertEqual(plat, self.eu.which_platform())

    def test_set_clipboard(self):
        expected = "xyzzy"
        GuiUtil.set_clipboard(expected)
        actual = GuiUtil.get_clipboard()
        logger.debug(f'Contents of clipboard: {actual}')
        self.assertEqual(expected, actual)

