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

    def init_options(self, proc_type: str, default: str = '<not assigned>') -> dict:
        """
        Initialize the dictionary of options for a given proc_type

        :param proc_type: SAS Proc type, e.g., 'MEANS'
        :param default:  str retrieved if key missing
        :return: empty dict: that defaults to default
        """
        ans = defaultdict(lambda: default)
        self._proc_options[proc_type] = ans
        return ans

    def add_option(self, proc_type: str, option: str, value: str) -> dict:
        """
        Add a given option / value (like 'DATA' / 'cars.data') to a proc_type (like 'MEANS')
        :param proc_type: Which dictionary to add to (e.g., 'MEANS', 'SQL')
        :param option: SAS option like 'BY' or 'DATA'
        :param value:  option value like 'mycol' or 'cars.data'
        :return:
        """
        this_dict = self._proc_options[proc_type]
        this_dict[option] = value
        return this_dict

    def get_option(self, proc_type: str, option: str) -> str:
        """
        For the given proc_type (like 'MEANS') retrieve the value of the given option.
        :param proc_type: Which dictionary to read from (e.g., 'MEANS', 'SQL')
        :param option: SAS option like 'BY' or 'DATA'
        :return: option value like 'mycol' or 'cars.data'
        """
        this_dict = self._proc_options[proc_type]
        return this_dict[option]