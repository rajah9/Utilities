import logging
from typing import List, Union, Callable
from re import match, search, sub, split
from string import ascii_letters
from urllib.parse import urlparse
from CollectionUtil import CollectionUtil
from pandas import DataFrame

Strings = List[str]
Cell = CollectionUtil.named_tuple('Cell', ['value', 'cellType'])

_DOUBLE_QUOTE = '"'
_SINGLE_QUOTE = "'"
_BACKSLASH = "\\"

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* Uses a generator in find_first_substring_in_list to find an element in a list.
* tests to see if something is a string with isinstance(field, str) in truncate.
* implements default dictionary to count words
* Uses a namedtuple to let Excel cells have both a value and a type.

Some notes about formatting:
* See https://docs.python.org/3/library/string.html for the definitive word. 
** s6 = '{lang} is over {0:0.1f} {date} years old.'.format(20, date='years', lang='Python')
      Python is over 20.0 years old
** print('{0:>8}'.format('101.55') 
         101.55   (field width of 8)
** print('{0:-^20}.format('hello'))
       ------hello-------
>>> for align, text in zip('<^>', ['left', 'center', 'right']):
...     '{0:{fill}{align}16}'.format(text, fill=align, align=align)
...
'left<<<<<<<<<<<<'
'^^^^^center^^^^^'
'>>>>>>>>>>>right'


'b' Binary format. Outputs the number in base 2.
'c' Character. Converts the integer to the corresponding unicode character before printing.
'd' Decimal Integer. Outputs the number in base 10.
'o' Octal format. Outputs the number in base 8.
'x' Hex format. Outputs the number in base 16, using lower-case letters for the digits above 9.
'n' Number. This is the same as 'g', except that it uses the current locale setting to insert the appropriate number separator characters.
'%' Percentage. Multiplies the number by 100 and displays in fixed ('f') format, followed by a percent sign.
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

    """
    Following classmethods can be called in lambda and apply functions.
    Example:
        f_str = StringUtil.capitalize_func('title') # could also be 'all-caps'
        self._df_inv = self.pu.replace_col_using_func(df=self._df_inv, column_name='Author', func=f_str)

    """

    @classmethod
    def capitalize_dict(cls) -> dict:
        """
        Return a dictionary whose keys are capitalization styles and whose values are string functions.
        :return:
        """
        str_funcs = {'title': cls.capitalize_as_title,
                     'all-caps': cls.all_caps,
                     'as-is': cls.capitalize_as_is,
                     'all-lower': cls.capitalize_all_lower}
        return str_funcs

    @classmethod
    def capitalize_func(cls, cap_style:str='as-is') -> Callable:
        try:
            f_str = cls.capitalize_dict()[cap_style]
        except KeyError:
            logger.warning(f'Unable to find capitalization style: {cap_style}')
            f_str = cls.capitalize_as_is
        return f_str

    @classmethod
    def all_caps(cls, myString:str) -> str:
        return myString.upper()

    @classmethod
    def capitalize_as_title(cls, myString:str) -> str:
        """
        capitalize as a title, so "for whom the bell tolls" -> "For Whom The Bell Tolls."
        :param myString: string to capitalize
        :return: First letter of each word capitalized
        """
        return myString.title()

    @classmethod
    def capitalize_as_is(cls, myString:str) -> str:
        """
        capitalize as is (basically a no-op). So "For Whom the bell tolls" -> "For Whom the bell tolls"
        :param myString: string to capitalize
        :return: as is.
        """
        return myString

    @classmethod
    def capitalize_all_lower(cls, myString:str) -> str:
        """
        make myString all lowercase, so "For Whom the Bell Tolls" -> "for whom the bell tolls"

        :param myString:
        :return: all lowercase
        """
        return myString.lower()

    @classmethod
    def find(cls, my_string: str, find_me: str, case_sensitive: bool = True) -> int:
        """
        Find find_me within my_string. If not found, return -1.
        :param my_string: (possibly empty) string to search
        :param find_me:
        :return: integer position (0-offset) of find_me, or -1 if not found.
        """
        haystack = my_string if case_sensitive else my_string.lower()
        needle = find_me if case_sensitive else find_me.lower()
        ans = haystack.find(needle) if haystack else -1
        return ans

    @classmethod
    def is_found(cls, my_string: str, find_me: str, case_sensitive: bool = True) -> bool:
        return StringUtil.find(my_string, find_me, case_sensitive) >= 0

    @classmethod
    def starts_with(cls, my_string: str, find_me: str, case_sensitive: bool = True) -> bool:
        """
        Return true if my_string starts with find_me.
        :param my_string:
        :param find_me:
        :return:
        """
        pos = StringUtil.find(my_string, find_me)
        return pos == 0

    @classmethod
    def reverse_find(cls, my_string: str, find_me: str, left_most: int = 0) -> int:
        """
        Find find_me within my_string, searching from the end and backwards. If not found, return -1.
        :param my_string:
        :param find_me:
        :param left_most: left-most index (or the last one to look at with rfind).
        :return: index of find_me (or -1 if not found)
        """
        return my_string.rfind(find_me, left_most)

    def capitalize_first_letter(self, myString:str=None) -> str:
        """
        Capitalize the first letter.
        :param self:
        :param myString:
        :return:
        """
        self.string = myString or self.string
        return self.string.capitalize()

    def capitalize_all_caps(self, myString:str=None) -> str:
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


    def split_on_delimiter(self, my_string: str = None, delim: str = ' ') -> list:
        """
        Split the given string up on the delimiter.
        :param my_string: string to split, like "A2:A5"
        :param delim: String with delimiter, like ":". Defaults to " "
        :return: list of strings, for example ["A2", "A5"]
        """
        self.string = my_string or self.string
        return self.string.split(delim)

    def split_on_regex(self, my_string: str = None, delim_regex: str = ' ', is_trim: bool = True) -> list:
        """
        Split given string on the given regex delimiters.
        :param my_string:
        :param delim_regex:
        :param is_trim: if true, trim leading and trailing blanks
        :return: list of strings
        """
        self.string = my_string or self.string
        ans = split(delim_regex, my_string)
        if is_trim:
            trimmed = [x.strip() for x in ans]
            return list(filter(None, trimmed))
        return ans

    def find_first_substring_in_list(self, my_string: str, my_list: list) -> str:
        """
        Search through my_list and return the first element containing my_string.
        :param my_string: string to search for
        :param my_list:  list of strings
        :return: first el containing my_string or None if not found.
        """
        return next((s for s in my_list if my_string in s), None)

    def find_first_prefix_in_list(self, my_string: str, my_list: list, default: str = 'Default', case_sensitive: bool = False) -> str:
        """
        Search through my_list and return the first element prefixed by my_string.
        :param my_string:
        :param my_list:
        :return:
        """
        for s in my_list:
            if StringUtil.starts_with(my_string=my_string, find_me=s, case_sensitive=case_sensitive):
                return s

        return default

    @staticmethod
    def find_substring_occurrences_in_list(my_string:str, my_list:List) -> List:
        """
        Return a list (possibly empty) of the occurrences of the substring my_string within my_list.
        :param my_string:
        :param my_list:
        :return: List of the occurrences of my_string within my_lsit.
        """
        return [s for s in my_list if my_string in s]

    def truncate(self, field:str, max:int=255) -> str:
        """
        truncate the given field to the max length
        :param field: field to truncate
        :param max: length to truncate to
        :return:
        """
        if isinstance(field, str):
            if len(field) < max:
                return field
            logger.warning(f'Truncating field: {field} to {max} characters.')
            return field[:max]
        else:
            logger.warning(f'Cannot truncate {field} because it is not a string.')
            return f'{field}'

    def leading_2_places(self, n:int) -> str:
        """
        Return the given integer with a leading 0 if it's less than 10.
        :param n: any non-negative int
        :return:
        """
        return f"{n:02d}"

    @staticmethod
    def regex_found(my_string:str, pattern:str=r".*") -> bool:
        """
        Return a boolean as to whether pattern is found within my_string.
        Helpful pattern examples:
          [\d,\.]+                  a number like 2,345.67
          r"[\d,\.]+\s+[\d,\.]"     Embedded space between two numbers
          ^[a-zA-Z]                 Starts with an alphabetic
        :param my_string: Look inside me.
        :param pattern: Pattern to look for, such as r"crime(s)?"
        :return: True iff the pattern was found.
        """
        ans = match(pattern, my_string)
        logger.debug(f'Results of {pattern} within {my_string} are: {ans}')
        return ans is not None

    @staticmethod
    def regex_position(my_string:str, pattern:str=r".*") -> int:
        """
        Return the position of the pattern within my)string.
        :param my_string: Look inside me.
        :param pattern: Pattern to look for, such as r"crime(s)?"
        :return: the start and end positions, or -1 if not found
        """
        s = search(pattern, my_string)
        if s:
            return s.start(), s.end()
        else:
            return -1, -1

    @staticmethod
    def surrounding_regex(my_string:str, pattern:str=r'.*', before_chars=5, after_chars=5) -> str:
        """
        Provide the regex found, plus or minus
        :param my_string: Look inside me.
        :param pattern: Pattern to look for, such as r"crime(s)?"
        :param before_chars: How many characters before the start to include.
        :param after_chars: How many characters after the end to include.
        :return: a substring including the extra characters.
        """
        start, end = StringUtil.regex_position(my_string, pattern)
        if start == -1:
            return None
        return my_string[max(start-before_chars, 0) : min(end+after_chars, len(my_string))]

    @staticmethod
    def replace_all_regex(my_string:str, regex:str, new:str) -> str:
        """
        Replace the regex pattern in my_string
        :param my_string: original string, like "liking it"
        :param regex: example "[Ll]iking" (may be nice to quote with an r with embedded blackslashes
        :param new: new string to substitute, like "loving"
        :return: new string
        """
        return sub(regex, new, my_string)

    def excel_col_to_int(self, my_string: str = None) -> int:
        """
        Convert Excel column A or B or AA to 1, 2, and 27.
        Modified from https://stackoverflow.com/a/12640614/509840
        :param my_string: Can be single like A or mixed like AB24. Not case sensitive.
        :return: integer representing the letter part
        """
        self.string = my_string or self.string
        num = 0
        for c in self.string:
            if c in ascii_letters:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num

    def int_to_excel_col(self, col_num: int) -> str:
        """
        Convert an integer like 1, 2, or 27 to an Excel column like A, B, or AA.
        :param col_num:
        :return:
        """
        def conv_num_to_letter(n: int) -> str:
            return chr(n +  ord('A') - 1)
        ans = ""
        if col_num <= 702: # 702 = 26*27 or ZZ
            second_num = col_num
            if col_num > 26:
                first_num = int((col_num - 1) / 26)
                ans = conv_num_to_letter(first_num)
                second_num = col_num - first_num * 26
            return ans + conv_num_to_letter(second_num)
        logger.warning(f'Columns only go up to ZZ (or 702), but {col_num} was requested. Returning "BAD"')
        return 'BAD'

    def digits_only(self, my_string: str = None) -> str:
        """
        convert the digits-only portion of a string like "AB14" to string "14" and "D32" to "32".
        :param my_string:
        :return:
        """
        self.string = my_string or self.string
        # Extract the numbers only. From https://www.geeksforgeeks.org/python-extract-digits-from-given-string/
        res = sub(r"\D", "", self.string)
        return res

    def digits_only_as_int(self, my_string = None) -> int:
        """
        Convert digits in myString to an int. If myString contains no digits, issue a warning and return 0.
        :param my_string:  String like "A25'
        :return: int like 25
        """
        self.string = my_string or self.string
        res = self.digits_only(self.string)
        if res:
            return int(res)
        logger.warning(f'String {my_string} had no digits, so returning 0.')
        return 0

    def substring_between_tokens(self, my_string: str = None, start_tok: str = None, end_tok: str = None) -> str:
        """
        Return the substring from start_str to end_str, excluding those tokens.
        :param my_string: string to search, e.g., "A (parenthetical) phrase"
        :param start_tok: token to start the search on, e.g., "(". If None, start from the beginning.
        :param end_tok: token to end the search on, such as ")". If None, go to the end.
        :return: substring between the tokens, e.g., parenthetical.
        """
        start_pos = 0
        if start_tok:
            start_pos = StringUtil.find(my_string, start_tok)
            if start_pos < 0:
                logger.warning(f'Token {start_tok} not found within {my_string}. Returning none.')
                return None
            start_pos += len(start_tok)
        end_pos = len(my_string)
        if end_tok:
            end_pos = StringUtil.reverse_find(my_string, end_tok)
            if end_pos < 0:
                logger.warning(f'Token {start_tok} not found within {my_string}. Returning none.')
                return None
        return my_string[start_pos:end_pos]

    def substring_inside_quotes(self, my_string: str) -> str:
        """
        Return the substring from within the quotes, either ' or "
        :param my_string: string to search, e.g., <I said, "Don't stop"
        :return: substring inside the two quotes, e.g., Don't stop
        """
        start_single = StringUtil.find(my_string, _SINGLE_QUOTE)
        start_double = StringUtil.find(my_string, _DOUBLE_QUOTE)
        if (start_double == -1) and (start_single == -1):
            # Neither quote found
            logger.debug(f'no quotes found in <{my_string}>. Returning string unchanged.')
            return my_string
        if (start_double == -1):
            # Only single found
            start_pos = start_single
            match_me = _SINGLE_QUOTE
        elif (start_single == -1):
            # Only double quote found
            start_pos = start_double
            match_me = _DOUBLE_QUOTE
        elif (start_single < start_double):
            # Single found first
            start_pos = start_single
            match_me = _SINGLE_QUOTE
        else:
            # Double quote found first
            start_pos = start_double
            match_me = _DOUBLE_QUOTE

        match_pos = StringUtil.reverse_find(my_string, match_me, left_most = start_pos + 1)
        if (match_pos == -1):
            logger.warning(f'Initial <{match_me} found in {my_string}, but matching quote not found. Returning None.')
            return None
        return my_string[start_pos + 1 : match_pos]

    def as_float_or_int(self, myString: str = None) -> Union[int, float]:
        """
        Convert digits in myString to an int. Failing that, try converting to a float. If that also fails, warn user and return 0.
        :param myString:  String like '25' or '3.14' or '1,234.56'
        :return: int like 25 or float like 3.14
        """
        no_commas = myString or self.string
        self.string = self.replace_all(',', '', no_commas)
        try:
            as_int = int(self.string)
            return as_int
        except ValueError:
            # Catch the ValueError, but do nothing, and try as a float
            pass

        try:
            as_float = float(self.string)
        except ValueError:
            logger.warning(f'String {myString} is not a float or an int, so returning 0.')
            return 0
        return as_float

    def surround_with_quotes(self, my_string: str) -> str:
        """
        Surround the string with (preferably) single quotes, possibly double.
          hello => 'hello'
          don't => "don't"
          I said, "don't do it." => I said, "don\'t do it."
        :param my_string:
        :return:
        """
        if not StringUtil.is_found(my_string, _SINGLE_QUOTE):
            # No single quotes found, so surround with single quotes
            return _SINGLE_QUOTE + my_string + _SINGLE_QUOTE
        elif not StringUtil.is_found(my_string, _DOUBLE_QUOTE):
            return _DOUBLE_QUOTE + my_string + _DOUBLE_QUOTE
        else:
            ans = self.replace_all(my_string=my_string, old=_SINGLE_QUOTE, new=_BACKSLASH+_SINGLE_QUOTE)
            return _SINGLE_QUOTE + ans + _SINGLE_QUOTE

    def convert_string_append_type(self, my_string: str = None) -> Cell:
        """
        Decide to leave the string as a string, or convert it to a number (or percentage).
        :param my_string: String like 'hello', '2,345.67', or '99.4%'
        :return: a named tuple with (hello, 'Normal'), (2345.67, 'Comma'), or (.994, 'Percent')
        """
        self.string = my_string or self.string
        if '%' in self.string:
            no_percent = self.replace_first(old='%', new='')
            val = self.as_float_or_int(no_percent) / 100.0 # might need to divide by 100
            return Cell(value=val, cellType='Percent')
        elif self.regex_found(pattern=r"[\d,\.]+", my_string=self.string):
            no_comma = self.replace_first(old=',', new='')
            val = self.as_float_or_int(no_comma)
            return Cell(value=val, cellType='Comma')
        else:
            return Cell(value=self.string, cellType='Normal')

    def nth_word(self, my_string: str = None, word_index: int = 1, delim: str = " ") -> str:
        """
        Return the nth word in my_string.
        :param my_string: words in a string
        :param word_index: if n=2, return the 2nd word (one-offset)
        :param delim: word separator, usually a space.
        :return: nth word, or None if it doesn't have that many words.
        """
        self.string = my_string or self.string
        words = self.string.split(delim)
        word_desired = word_index - 1
        if word_index <= len(words):
            return words[word_desired].strip()
        logger.warning(f'You asked for word {word_index}, but {my_string} does not have that many words. Returning None.')
        return None

    def remove_single_line_comment(self, my_string: str, start_comment: str = '/*', end_comment: str = '*/', is_trim: bool = True) -> str:
        """
        Remove a single-line comment, so that "a += 1 /* Increment a */" becomes a + 1
        :param line:
        :param start_comment:
        :param end_comment:
        :param is_trim: True if you want to remove trailing and leading blanks
        :return:
        """
        self.string = my_string or self.string
        start_pos = StringUtil.find(self.string, find_me=start_comment)
        end_pos = StringUtil.reverse_find(self.string, find_me=end_comment)
        if start_pos >= 0 and end_pos >= 0:
            comment = self.string[start_pos:end_pos + len(end_comment)]
            ans = self.replace_first(old=comment, new="", myString=my_string)
            return ans.strip() if is_trim else ans
        logger.debug(f'Did not find both opening and closing comment markers in <{my_string}>. Returning original string')
        return my_string

    def remove_comments(self, lines: Strings, start_comment: str = '/*', end_comment: str = '*/', is_trim: bool = True) -> Strings:
        """
        Looking at all the lines, remove all comments between start_comment and stop_comment tokens
        :param lines:
        :param start_comment:
        :param end_comment:
        :return:
        """
        ans = []
        in_multiline_comment = False
        for line in lines:
            has_comment_open = (start_comment in line)
            has_comment_close = (end_comment in line)
            if has_comment_open and has_comment_close:
                ans.append(self.remove_single_line_comment(my_string=line, start_comment=start_comment, end_comment=end_comment, is_trim=is_trim))
            elif has_comment_open:
                # Start of a multi-line comment
                start_pos = StringUtil.find(my_string=line, find_me=start_comment)
                if start_pos > 0:
                    # It's not in the first position, so add the part up to start_pos
                    code = line[:start_pos].strip() if is_trim else line[:start_pos]
                    ans.append(code)
                in_multiline_comment = True
            elif has_comment_close and in_multiline_comment:
                # End of a multi-line comment
                end_pos = StringUtil.find(my_string=line, find_me=end_comment)
                if end_pos >= 0:
                    # It's not in the first position, so add the part up to start_pos
                    code = line[end_pos + len(end_comment):].strip() if is_trim else line[end_pos + len(end_comment):]
                    if code:
                        ans.append(code)
                in_multiline_comment = False
            elif in_multiline_comment:
                pass
            else:
                ans.append(line)
        return ans

    def is_variable(self, my_string: str = None) -> bool:
        """
        Test to see if my_string looks like an identifier.
        :param my_string: string like valid_var2 or 2invalid%
        :return: True iff it looks like an identifier.
        """
        self.string = my_string or self.string
        return StringUtil.regex_found(self.string, pattern=r"^[^\d\W]\w*\Z")

    def is_SAS_variable(self, my_string: str = None) -> bool:
        """
        Test to see if my_string looks like an identifier.
        This was derived from is_variable, but includes not just one23 but also &one23.
        :param my_string: string like valid_var2 or 2invalid%
        :return: True iff it looks like a SAS identifier.
        """
        self.string = my_string or self.string
        return StringUtil.regex_found(self.string, pattern=r"^([^\d\W]\w*)|(&[^\d\W]\w*\.)\Z")

    def extract_variables(self, rhs: str) -> Strings:
        """
        For an assignment like
          xyz = min(a, b, c)
        return the rhs as a list, e.g., ['min', 'a', 'b', 'c']
        :param rhs:
        :return: list of variable identifiers
        """
        ans = self.split_on_regex(my_string=rhs, delim_regex="\\*{1,2}|\\+| |\\/|\\-|\\(|\\)|,")
        return [x for x in ans if self.is_variable(x)]

    def parse_url(self, url: str = None) -> dict:
        """
        Parse the URL into sections, such as scheme, netloc, path, and params.
        :param url:
        :return:
        """
        self.string = url or self.string
        return urlparse(self.string)

    def fill_string(self, my_str: str, fill_str: str = '*', fill_width: int = 80, alignment: str = 'center') -> str:
        """
        Print out my_str, surrounded by fill_str, to a width of fill_width.
        :param my_str: string to print out
        :param fill_str: fill string (default is '*')
        :param fill_width: how wide to make the string (default 80)
        :param alignment: 'center', 'left', or 'right' aligned
        :return:
        """
        align_dict = {'center': '^', 'left': '<', 'right': '>'}
        try:
            align = align_dict[alignment.lower()]
        except KeyError:
            valid_keys = align_dict.keys()
            logger.debug(f"Alignment should be one of: {','.join(valid_keys)} but is {alignment}. Setting to center.")
            align = align_dict['center']

        return '{0:{fill}{align}{width}}'.format(my_str, fill=fill_str, align=align, width=fill_width)

    def get_prefix_in_list(self, my_str: str, allowed_prefix: Strings = ['PRE', 'POST'], default: str = 'Other',
                           case_sensitive=False, must_cap_ret_val: bool = False) -> str:
        """
        Take a string (like Postal) and a list of prefixes like ['PRE', 'POST'] and return the matching prefix (or default)
        :param my_str:
        :param allowed_prefix:
        :param default:
        :param case_sensitive:
        :param must_cap_ret_val:
        :return:
        """
        ans = self.find_first_prefix_in_list(my_str, allowed_prefix, None, case_sensitive)
        if not ans:
            return default
        if must_cap_ret_val:
            return ans.upper()

        return ans

"""
This class accumulates lines (say, for a log).
"""

class LineAccumulator:
    def __init__(self):
        self._contents = []

    # Getter for content
    @property
    def contents(self):
        return self._contents if self._contents else []

    @contents.setter
    def contents(self, lines: Strings):
        self._contents = lines

    def add_line(self, line: str):
        self._contents.append(line)

    def add_lines(self, lines: Strings):
        self._contents.extend(lines)

    def add_line_or_lines(self, l: Union[str, Strings]):
        if isinstance(l, list):
            self.add_lines(l)
        elif isinstance(l, str):
            self.add_line(l)
        else:
            logger.warning(f'Only str or list of strings accepted, but got {type(l)}')
        return

    def add_df(self, df:DataFrame, how_many_rows: int = 10):
        ans = str(df.head(how_many_rows))
        lines = ans.splitlines()
        self.add_lines(lines)

    def contents_len(self):
        return len(self._contents)





