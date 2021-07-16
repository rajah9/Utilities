import logging
from unittest import TestCase

from Add_path import Add_path

Add_path.add_parent()
from LogitUtil import logit
from StringUtil_specialized import StringUtil_specialized

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Following class separates some of the obscure, one-off StringUtil routines,
such as handling phone numbers or fixing upper-ascii. 
These were broken out because it's not often necessary to import ftfy or phonenumbers. 
"""

class TestStringUtil_specialized(TestCase):
    def setUp(self):
        logger.debug('Starting TestStringUtil_specialized')
        self.su = StringUtil_specialized()

    @logit()
    def test_parse_phone(self):
        phones = ['800-328-1452', '8774467746', '-   -   0']
        expected = ['(800)328-1452', '(877)446-7746', '']
        for i, phone in enumerate(phones):
            self.assertEqual(expected[i], self.su.parse_phone(phone))
        self.assertEqual('(800) 328-1452', self.su.parse_phone('800-328-1452', should_remove_blanks=False))

    @logit()
    def test_fix_text(self):
        base = '\ufeffParty like it&rsquo;s 1999!'
        expected = "Party like it's 1999!"
        self.assertEqual(expected, self.su.fix_text(base))

