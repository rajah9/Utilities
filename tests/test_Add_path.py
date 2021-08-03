import logging
from pathlib import Path
from unittest import TestCase

from Add_path import Add_path

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Test_Add_path(TestCase):
    def test_find_ancestor_with_child(self):
        # Test 1. Normal. Use real paths on this system.
        cwd = Path().resolve()
        parents = cwd.parents
        for lvl, parent in enumerate(parents):
            subdirs = [d for d in parent.iterdir() if d.is_dir()]
            if subdirs:
                last_subdir = subdirs.pop()
                logger.debug(f'parent {parent} (level {lvl}) has a subdir {last_subdir}')
                act1 = Add_path.find_ancestor_with_child(child=last_subdir)
                self.assertIsNotNone(act1, f'fail test 1: parent {parent} has a subdir {last_subdir}')
        # Test 2. Normal. Use real paths on this system, but from the root to cwd.
        rev_parents = cwd.parents.__reversed__()
        for lvl, parent in enumerate(rev_parents):
            subdirs = [d for d in parent.iterdir() if d.is_dir()]
            if subdirs:
                first_subdir = subdirs.pop(0)
                logger.debug(f'parent {parent} (level {lvl}) has a subdir {first_subdir}')
                act2 = Add_path.find_ancestor_with_child(child=first_subdir, search_from_leaf=False)
                self.assertIsNotNone(act2, f'fail test 2: parent {parent} has a subdir {first_subdir}')
