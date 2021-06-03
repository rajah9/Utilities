import sys
from pathlib import Path
import logging
from unittest import mock, TestCase, main
from CompilerUtil import SasCompilerUtil
from LogitUtil import logit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
"""

_PROC_TYPE = 'MEANS'
class Test_CompilerUtil(TestCase):
    def setUp(self):
        self.scu = SasCompilerUtil()

    @logit()
    def test_proc_options(self):
        # Test 1. Initialize proc options. The dictionary returned must be the same as the property.
        d = self.scu.init_options()
        self.assertEqual(d, self.scu.proc_options)

    @logit()
    def test_init_options(self):
        # Test 1. Initialize options and retrieve a default
        test_default = 'Not initialized'
        d = self.scu.init_options(default=test_default)
        missing_val = d['noSuchKey']
        self.assertEqual(test_default, missing_val)
        # Test 2. Make the default None.
        test_default = None
        d = self.scu.init_options(default=test_default)
        missing_val = d['doNotHaveThisKeyEither']
        self.assertIsNone(missing_val)
        # Test 3. If we don't set the default, we should get '<not assigned>'
        d = self.scu.init_options()
        self.assertEqual('<not assigned>', d['nopeNotThisOne'])

    @logit()
    def test_add_option(self):
        self.scu.init_options(_PROC_TYPE)
        test1_opt = 'DATA'
        test1_val = 'cars.data'
        act = self.scu.add_option(option=test1_opt, value=test1_val)
        self.assertEqual(test1_val, act[test1_opt])

    @logit()
    def test_get_option(self):
        self.scu.init_options(_PROC_TYPE)
        test1_opt = 'DATA'
        test1_val = 'cars.data'
        self.scu.add_option(option=test1_opt, value=test1_val)
        act = self.scu.get_option(option=test1_opt)
        self.assertEqual(test1_val, act)

if __name__ == '__main__':
    main()
