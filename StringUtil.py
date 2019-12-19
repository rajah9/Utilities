import logging
import ftfy
import phonenumbers

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
"""
Interesting Python features:
* Uses a generator in find_first_substring_in_list to find an element in a list.

"""
class StringUtil:
    def __init__(self, myString:str='Uninitialized'):
        self.string = myString

    # Getters and setters for string
    @property
    def string(self):
        return self._string
    # Setter for string.
    @string.setter
    def string(self, s:str):
        self._string = s

    def capitalize_first_letter(self, myString:str=None) -> str:
        """
        Capitalize the first letter.
        :param self:
        :param myString:
        :return:
        """
        self.string = myString or self.string
        return self.string.capitalize()

    def capitalize_as_title(self, myString:str=None) -> str:
        """
        capitalize as a title, so "for whom the bell tolls" -> "For Whom The Bell Tolls."
        :param myString: string to capitalize
        :return: First letter capitalized
        """
        self.string = myString or self.string
        return self.string.title()

    def capitalize_as_is(self, myString:str=None) -> str:
        """
        capitalize as is (basically a no-op). So "For Whom the bell tolls" -> "For Whom the bell tolls"
        :param myString: string to capitalize
        :return: as is.
        """
        self.string = myString or self.string
        return self.string


    def all_caps(self, myString:str=None) -> str:
        """
        make myString all capitals, so "for whom the bell tolls" -> "FOR WHOM THE BELL TOLLS"

        :param myString:
        :return:
        """
        self.string = myString or self.string
        return self.string.upper()

    def first_non_blank_letter(self, myString:str=None) -> str:
        """
        Return the first non-blank letter.
        :param myString:
        :return:
        """
        self.string = myString or self.string
        return self.string.lstrip()[0]

    def replace_n(self, old:str, new:str, n:int, myString:str=None) -> str:
        """
        Replace the first n occurrences of old in myString with the new.
        If n is 0 or negative, replace all.
        Warn if the string was not found and return the original string.
        :param old:
        :param new:
        :param n:
        :param myString:
        :return:
        """
        self.string = myString or self.string
        ans = self.string
        try:
            if n < 1:
                ans = self.string.replace(old, new)
            else:
                ans = self.string.replace(old, new, n)
        except ValueError:
            logger.warning(f'String {old} not found in {ans}, so returning original string.')
        finally:
            return ans

    def replace_first(self, old:str, new:str, myString:str=None) -> str:
        """
        Replace the first occurrence of old in myString with the new.
        Warn if the string was not found and return the original string.
        :param old:
        :param new:
        :param myString:
        :return:
        """
        return self.replace_n(old, new, 1, myString)

    def replace_all(self, old:str, new:str, myString:str=None) -> str:
        """
        Replace all occurrences of old in myString with the new.
        Warn if the string was not found and return the original string.
        :param old:
        :param new:
        :param myString:
        :return:
        """
        return self.replace_n(old, new, 0, myString)

    def is_vowel(self, myString:str=None) -> bool:
        """
        Return true if and only if the first character in myString is a vowel.
        :param myString:
        :return:
        """
        self.string = myString or self.string
        return self.string.lower() in {'a', 'e', 'i', 'o', 'u'}

    def add_affixes(self, myString:str=None, prefix:str='', suffix=''):
        """
        Return myString with the given prefix (if any) and suffix (if any).
        :param myString:
        :param prefix:
        :param suffix:
        :return:
        """
        self.string = myString or self.string
        return prefix + self.string + suffix

    def regex_prefix(self, myString:str=None, is_case_insensitive:bool=True, is_single_line_mode:bool=False):
        """
        Build a regex mode string, such as (?i) or (?is).
        :param is_case_insensitive:
        :param is_single_line_mode:
        :return:
        """
        ans = '(?'
        if is_case_insensitive:
            ans = ans + 'i'
        if is_single_line_mode:
            ans = ans + 's'
        return ans + ')'

    def fix_text(self, my_string:str=None) -> str:
        """
        Fixes the text with funny characters.
        Documentation at https://ftfy.readthedocs.io/en/latest
        :param my_string:
        :return:
        """
        ans = ftfy.fix_text(my_string)
        return ans

    def find_first_substring_in_list(self, my_string:str, my_list:list) -> str:
        """
        Search through my_list and return the first element containing my_string.
        :param my_string: string to search for
        :param my_list:  list of strings
        :return: first el containing my_string or None if not found.
        """
        return next((s for s in my_list if my_string in s), None)

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

    def truncate(self, field:str, max:int=255) -> str:
        """
        truncate the given field to the max length
        :param field: field to truncate
        :param max: length to truncate to
        :return:
        """
        if len(field) < max:
            return field
        logger.warning(f'Truncating field: {field} to {max} characters.')
        return field[:max]

    def leading_2_places(self, n:int) -> str:
        """
        Return the given integer with a leading 0 if it's less than 10.
        :param n: any non-negative int
        :return:
        """
        return f"{n:02d}"