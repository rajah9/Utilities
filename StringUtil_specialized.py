import logging

import ftfy
import phonenumbers
from pandas import DataFrame

from StringUtil import StringUtil

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Following class separates some of the obscure, one-off StringUtil routines,
such as handling phone numbers or fixing upper-ascii. 
These were broken out because it's not often necessary to import ftfy or phonenumbers. 
"""

class StringUtil_specialized(StringUtil):
    def fix_text(self, my_string:str=None) -> str:
        """
        Fixes the text with funny characters.
        Documentation at https://ftfy.readthedocs.io/en/latest
        :param my_string:
        :return:
        """
        ans = ftfy.fix_text(my_string)
        return ans

    def parse_phone(self, phone:str, should_remove_blanks:bool = True) -> str:
        try:
            num = phonenumbers.parse(phone, "US")
            formatted_num = phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.NATIONAL)
            if should_remove_blanks:
                return self.replace_all(old=" ", new="", myString=formatted_num)
            else:
                return formatted_num
        except phonenumbers.phonenumberutil.NumberParseException:
            logger.warning(f'could not parse number <{phone}>. Returning blank.')
            return ""
