from collections import namedtuple
from YamlUtil import YamlUtil
from tests.test_FileUtil import Test_FileUtil
import logging
from LogitUtil import logit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features.
* TestYamlUtil is a subclass of Test_FileUtil, paralleling the classes under test.
* Uses super to call the parent's __init__ member.
* Uses setUp to create a file for every test.
* Uses zip to create a dict from two lists.
"""
class TestYamlUtil(Test_FileUtil):
    keys = ['firstname', 'lastname', 'zip']
    vals = ['Rajah', 'Chacko', 28269]
    expected_dict = dict(zip(keys, vals))

    def __init__(self, *args, **kwargs):
        super(TestYamlUtil, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.create_yaml(self.keys, self.vals)

    @logit()
    def test_asdict(self):
        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        d = YamlUtil(qualifiedPath)
        self.assertEqual(d.asdict, self.expected_dict)

    @logit()
    def test_asnamedtuple(self):
        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        d = YamlUtil(qualifiedPath)
        actual = d.asnamedtuple
        Expected_tuple = namedtuple('yaml_helper', self.keys)
        expected_tuple = Expected_tuple(**self.expected_dict)
        self.assertEqual(expected_tuple, actual)
        for k, v in self.expected_dict.items():
            item = getattr(expected_tuple, k)
            self.assertEqual(item, v)

    @logit()
    def test_fields(self):
        qualifiedPath = self._fu.qualified_path(self.path, self.yaml)
        d = YamlUtil(qualifiedPath)
        actual = d.fields
        self.assertEqual(actual, self.keys)



