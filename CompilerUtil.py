import logging
from collections import defaultdict

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CompilerUtil:
    """
    Base class for a compiler.
    This is paired with ParserUtil.
    """

    def __init__(self, **kw):
        self.names = []

class SasCompilerUtil(CompilerUtil):
    def __init__(self, **kw):
        self._proc_options = {}

    @property
    def proc_options(self):
        return self._proc_options

    def init_options(self, default: str = '<not assigned>') -> dict:
        """
        Initialize the dictionary of options for a given proc_type

        :param default:  str retrieved if key missing
        :return: empty dict: that defaults to default
        """
        ans = defaultdict(lambda: default)
        self._proc_options = ans
        return ans

    def add_option(self, option: str, value: str) -> dict:
        """
        Add a given option / proc_type (like 'DATA' / 'cars.data') to the options dictionary.
        :param option: SAS option like 'BY' or 'DATA'
        :param value:  option proc_type like 'mycol' or 'cars.data'
        :return:
        """
        self._proc_options[option] = value
        return self._proc_options

    def get_option(self, option: str) -> str:
        """
        Retrieve the proc_type of the given option.
        :param option: SAS option like 'BY' or 'DATA'
        :return: option proc_type like 'mycol' or 'cars.data'
        """
        return self._proc_options[option]