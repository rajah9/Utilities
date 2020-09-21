import unittest
import logging
from LogitUtil import logit
from StringUtil import StringUtil, LineAccmulator
from pandas import DataFrame
from CollectionUtil import CollectionUtil

_SINGLE_QUOTE = "'"

_DOUBLE_QUOTE = '"'

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
* Binds a class function in test_capitalize_func. See answer at https://stackoverflow.com/a/114289/509840
"""

class TestStringUtil(unittest.TestCase):
    def setUp(self):
        logger.debug('Starting TestStringUtil')
        self.su = StringUtil()

    @logit()
    def test_capitalize_dict(self):
        d = StringUtil.capitalize_dict()
        f_expected = StringUtil.capitalize_as_title
        self.assertEqual(f_expected, d['title'])

    @logit()
    def test_capitalize_func(self):
        original = "This is a string"
        f_expected = StringUtil.capitalize_all_caps
        f_actual = StringUtil.capitalize_func('all-caps')
        # Following bound_function because f_actual is unbound.
        c = StringUtil()
        bound_function = StringUtil.__dict__[f_actual.__name__].__get__(c, StringUtil)
        logger.debug(f'Bound function is: {bound_function}')
        self.assertEqual(bound_function, f_actual, 'Should point to same function')

        f_actual = StringUtil.capitalize_func('nosuchfeature') # should give as-is
        expected = f_actual(original)
        self.assertEqual(expected, original)

    def test_reverse_find(self):
        # Test 1. Normal
        test1 = "Madam, I'm Adam."
        find_me = "am."
        exp1 = test1.find(find_me, 10)
        self.assertEqual(exp1, StringUtil.reverse_find(test1, find_me))

    def test_all_caps(self):
        original = "for whom the bell tolls"
        expected = "FOR WHOM THE BELL TOLLS"
        self.assertEqual(expected, StringUtil.all_caps(original))

    @logit()
    def test_capitalize_first_letter(self):
        test1 = "this is a string"
        exp1 = "This is a string"
        self.assertEqual(exp1, self.su.capitalize_first_letter(test1))
        # Should work if already capitalized
        self.assertEqual(exp1, self.su.capitalize_first_letter(exp1))
        # Should work for an empty string
        empty = "  "
        self.assertEqual(empty, self.su.capitalize_first_letter(empty))

    @logit()
    def test_capitalize_as_title(self):
        test1 = "this is a string"
        exp1 = "This Is A String"
        self.assertEqual(exp1, self.su.capitalize_as_title(test1))

    @logit()
    def test_capitalize_all_lower(self):
        test = "For Whom the Bell Tolls"
        expected = "for whom the bell tolls"
        self.assertEqual(expected, self.su.capitalize_all_lower(test))

    @logit()
    def test_capitalize_all_caps(self):
        original = "for whom the bell tolls"
        expected = "FOR WHOM THE BELL TOLLS"
        self.assertEqual(expected, self.su.capitalize_all_caps(original))

    @logit()
    def test_capitalize_as_is(self):
        expected = "For Whom the bell tolls"
        self.assertEqual(expected, self.su.capitalize_as_is(expected))
    @logit()
    def test_first_non_blank_letter(self):
        test1 = "this is a string"
        exp1 = "t"
        self.assertEqual(exp1, self.su.first_non_blank_letter(test1))
        test2 = "  this"
        exp2 = "t"
        self.assertEqual(exp2, self.su.first_non_blank_letter(test2))

    @logit()
    def test_replace_first(self):
        test1 = "I am what I am."
        exp1 = "You are what I am."
        self.assertEqual(exp1, self.su.replace_first("I am", "You are", test1))
        # Test for string not found
        self.assertEqual(test1, self.su.replace_first("xyzzy", "wasn't found", test1))

    @logit()
    def test_replace_all(self):
        test1 = "I am what I am."
        exp1 = "You are what You are."
        self.assertEqual(exp1, self.su.replace_all("I am", "You are", test1))

    @logit()
    def test_replace_n(self):
        self.su.string = "Th* fiv* boxing wizards jump quickly. Som* mor* l*tt*r *."
        # Replace just the first occurrence.
        exp1 = "The fiv* boxing wizards jump quickly. Som* mor* l*tt*r *."
        self.assertEqual(exp1, self.su.replace_n("*", "e", 1))
        # Replace first three occurrences.
        exp2 = "The five boxing wizards jump quickly. Some mor* l*tt*r *."
        self.assertEqual(exp2, self.su.replace_n("*", "e", 3))
        # Replace all occurrences
        exp3 = "The five boxing wizards jump quickly. Some more letter e."
        self.assertEqual(exp3, self.su.replace_n("*", "e", -1))
        # Can't find the string to be replaced.
        exp4 = self.su.string
        self.assertEqual(exp4, self.su.replace_n("%", "e", -1))

    @logit()
    def test_is_vowel(self):
        self.su.string = "The string"
        self.assertFalse(self.su.is_vowel())
        self.su.string = "u"
        self.assertTrue(self.su.is_vowel())

    @logit()
    def test_add_affixes(self):
        myOriginal = 'Hello'
        suffix = ', World!'
        su = StringUtil(myOriginal)
        self.assertEqual('Hello, World!', su.add_affixes(suffix=suffix))

    def test_regex_prefix(self):
        su = StringUtil()
        self.assertEqual('(?i)', su.regex_prefix())
        self.assertEqual('(?is)', su.regex_prefix(is_case_insensitive=True, is_single_line_mode=True))

    @logit()
    def test_fix_text(self):
        base = '\ufeffParty like it&rsquo;s 1999!'
        expected = "Party like it's 1999!"
        self.assertEqual(expected, self.su.fix_text(base))

    def test_split_on_delimiter(self):
        # Test 1. Split on blanks by default.
        exp1 = ["hello", "there", "world"]
        test1 = ' '.join(exp1)
        self.assertEqual(exp1, self.su.split_on_delimiter(test1), 'failed test 1')
        # Test 2. Split on : for a range
        exp2 = ["A2", "A9"]
        test2 = ':'.join(exp2)
        self.assertEqual(exp2, self.su.split_on_delimiter(test2, delim=':'), 'failed test 2')

    def test_split_on_regex(self):
        # Test 1. Whitespace
        exp1 = ["To", "be", "or", "not", "to", "be"]
        test1 = "To be, or not to be"
        self.assertListEqual(exp1, self.su.split_on_regex(my_string=test1, delim_regex=r"\W"))
        self.assertListEqual(exp1, self.su.split_on_regex(my_string=test1, delim_regex=r"\W+", is_trim=False))

    @logit()
    def test_find_first_substring_in_list(self):
        l = ['abc123', 'def234', 'abc456']
        self.assertIsNone(self.su.find_first_substring_in_list('notInList', l))
        self.assertEqual(l[0], self.su.find_first_substring_in_list('abc', l))
        self.assertEqual(l[2], self.su.find_first_substring_in_list('abc4', l))

    @logit()
    def test_find_substring_occurrences_in_list(self):
        l = ['abc123', 'def234', 'abc456']
        # Test 1, normal case
        find_me = 'abc'
        actual = StringUtil.find_substring_occurrences_in_list(my_string=find_me, my_list=l)
        expected = []
        for s in filter(lambda x: find_me in x, l): expected.append(s)
        self.assertListEqual(expected, actual)
        # Test 2, empty case
        find_me = 'not in list'
        actual = StringUtil.find_substring_occurrences_in_list(my_string=find_me, my_list=l)
        expected = []
        self.assertListEqual(expected, actual)

    @logit()
    def test_parse_phone(self):
        phones = ['800-328-1452', '8774467746', '-   -   0']
        expected = ['(800)328-1452', '(877)446-7746', '']
        for i, phone in enumerate(phones):
            self.assertEqual(expected[i], self.su.parse_phone(phone))
        self.assertEqual('(800) 328-1452', self.su.parse_phone('800-328-1452', should_remove_blanks=False))

    @logit()
    def test_leading_2_places(self):
        self.assertEqual("02", self.su.leading_2_places(2))
        self.assertEqual("117", self.su.leading_2_places(117))
        self.assertEqual("99", self.su.leading_2_places(99))

    @logit()
    def test_truncate(self):
        # Test 1, normal.
        longer = '*' * 50
        my_max = 24
        truncated = '*' * my_max
        self.assertEqual(truncated, self.su.truncate(longer, max=my_max))
        # Test 2, max is > length
        my_max = len(longer) + 1
        self.assertEqual(longer, self.su.truncate(longer, max=my_max))
        # Test 3, not a string
        test3 = 1234
        exp3 = "1234"
        self.assertEqual(exp3, self.su.truncate(test3))

    @logit()
    def test_regex_found(self):
        # test 1
        text = "Financial crimes manager"
        pattern = r'[Ff]inancial [Cc]rimes'
        self.assertTrue(StringUtil.regex_found(text, pattern))
        # test 2
        should_not_find = "Financial manager"
        self.assertFalse(StringUtil.regex_found(should_not_find, pattern))

    @logit()
    def test_regex_position(self):
        # test 1
        text = "Financial crimes manager"
        pattern = "man"
        actual_start, actual_end = StringUtil.regex_position(text, pattern)
        expected_start = str.find(text, pattern)
        self.assertEqual(expected_start, actual_start)
        self.assertEqual(expected_start+ len(pattern), actual_end)
        # test 2
        should_not_find = "Wrong capitalization for Man"
        actual_start, actual_end = StringUtil.regex_position(should_not_find, pattern)
        self.assertEqual(-1, actual_start)
        self.assertEqual(-1, actual_end)

    @logit()
    def test_surrounding_regex(self):
        core = "found"
        surround_5 = "12345" + core + "54321"
        # test 1, normal
        actual_0 = StringUtil.surrounding_regex(surround_5, core, before_chars=0, after_chars=0)
        self.assertEqual(actual_0, core)
        # test 2, normal
        actual_5 = StringUtil.surrounding_regex(surround_5, core, before_chars=5, after_chars=5)
        self.assertEqual(actual_5, surround_5)
        # test 3, go beyond the bounds of a given string.
        actual_9 = StringUtil.surrounding_regex(surround_5, core, before_chars=9, after_chars=12)
        self.assertEqual(actual_9, surround_5)
        # test 4, find a substring right at the beginning.
        actual_begin = StringUtil.surrounding_regex(core, core)
        self.assertEqual(actual_begin, core)
        # test 5, regex pattern is not found
        test5 = "Normal"
        regex5 = "x|z"
        self.assertIsNone(self.su.surrounding_regex(my_string=test5, pattern=regex5))

    @logit()
    def test_replace_all_regex(self):
        # Test 1, remove white space and replace with !
        text = " Some  extra  Spaces Jim "
        pattern = r'\W+'
        new = "!"
        no_spaces = text.replace(' ', new)
        expected = no_spaces.replace("!!", "!")
        self.assertEqual(expected, StringUtil.replace_all_regex(my_string=text, regex=pattern, new=new))

    def test_find(self):
        # Test 1. Normal
        test1 = "xyzzy"
        find_me = "zy"
        exp1 = test1.find(find_me)
        self.assertEqual(exp1, StringUtil.find(test1, find_me), "failed test 1")
        # Test 2. Not found
        find_me = "NotHere"
        exp2 = -1
        self.assertEqual(exp2, StringUtil.find(test1, find_me), "failed test 2")

    def test_substrings_between_tokens(self):
        # Test 1. Normal.
        exp1 = "parenthetical"
        start_tok = "("
        end_tok = ")"
        test1 = "Find the " + start_tok + exp1 + end_tok + " phrase"
        self.assertEqual(exp1, self.su.substring_between_tokens(test1, start_tok=start_tok, end_tok=end_tok), "failed test 1")
        # Test 2. start_tok is None
        test2 = "Take it to the limit." # Note there are two "it" and we want the last.
        end_tok = "it"
        exp2 = "Take it to the lim"
        self.assertEqual(exp2, self.su.substring_between_tokens(test2, start_tok=None, end_tok=end_tok), "failed test 2")
        # Test 3. end_tok is None.
        test3 = "Take it to the limit."
        start_tok = "the "
        end_tok = None
        exp3 = "limit."
        self.assertEqual(exp3, self.su.substring_between_tokens(test3, start_tok=start_tok, end_tok=end_tok), "failed test 3")
        # Test 4. Both start_tok and end_tok are None.
        test4 = "Take it to the limit." # Note there are two "it" and we want the last.
        start_tok = None
        end_tok = None
        exp4 = test4
        self.assertEqual(exp4, self.su.substring_between_tokens(test4), "failed test 4")
        # Test 5. Not found
        test5 = "Take it to the limit." # Note there are two "it" and we want the last.
        start_tok = "nOt FoUnD"
        self.assertIsNone(self.su.substring_between_tokens(test5, start_tok=start_tok), "failed test 5")

    def test_substring_inside_quotes(self):
        # Test 1, normal
        exp1 = "Don't stop"
        test1 = "I said" + _DOUBLE_QUOTE + exp1 + _DOUBLE_QUOTE
        self.assertEqual(exp1, self.su.substring_inside_quotes(test1), "failed test 1")
        # Test 2, no quotes
        test2 = exp2 = "This has no quotes."
        self.assertEqual(exp2, self.su.substring_inside_quotes(test2), "failed test 2")
        # Test 3, one unmatched quote
        test3 = "Don't stop"
        self.assertIsNone(self.su.substring_inside_quotes(test3), "failed test 3")
        # Test 4, embedded quotes
        exp4 = "Judy said" + _DOUBLE_QUOTE + exp1 + _DOUBLE_QUOTE
        test4 = "I recall her saying, " + _SINGLE_QUOTE + exp4 + _SINGLE_QUOTE + "."
        self.assertEqual(exp4, self.su.substring_inside_quotes(test4), "failed test 4")

    def test_as_float_or_int(self):
        # Test 1, int
        exp1 = 20201225
        test1 = str(exp1)
        self.assertEqual(exp1, self.su.as_float_or_int(test1), "failed test 1")
        # Test 2, float
        exp2 = 3.14159
        test2 = "3.14159"
        self.assertEqual(exp2, self.su.as_float_or_int(test2), "failed test 2")
        # Test 3, bad string
        test3 = "neither int nor float"
        exp3 = 0
        self.assertEqual(exp3, self.su.as_float_or_int(test3), "failed test 3")
        # Test 4, float with a comma
        test4 = "1,234.56"
        exp4 = 1234.56
        self.assertEqual(exp4, self.su.as_float_or_int(test4), "failed test 4")

    def test_nth_word(self):
        # Test 1, normal (last word)
        exp1 = "last"
        test1 = "Pick out the last"
        self.assertEqual(exp1, self.su.nth_word(my_string=test1, word_index=4))
        # Test 2, normal (first word)
        exp2 = "Pick"
        test2 = "Pick up the first"
        self.assertEqual(exp2, self.su.nth_word(my_string=test2, word_index=1))
        # Test 3, error
        self.assertIsNone(self.su.nth_word(my_string=test2, word_index=5))

    def test_remove_single_line_comment(self):
        # Test 1. normal.
        exp1 = "a = 1"
        test1 = exp1 + "/* comment at end of line */"
        self.assertEqual(exp1, self.su.remove_single_line_comment(test1, is_trim=False), "failed test 1")
        # Test 2. With the trim.
        exp2 = " a = 1    "
        test2 = exp2 + "/* comment at end of line */"
        self.assertEqual(exp2.strip(), self.su.remove_single_line_comment(test2, is_trim=True), "failed test 2")
        # Test 3. Found an opening but not a closing
        test3 = exp3 = "Unbalanced comment /* stuff"
        self.assertEqual(exp3, self.su.remove_single_line_comment(test3), "failed test 3")

    def test_remove_comments(self):
        # Test 1. Single line comment
        code1 = "a = 1"
        test1 = code1 + "/* comment at end of line */"
        exp1 = [code1]
        self.assertEqual(exp1, self.su.remove_comments([test1]), "failed test 1")
        # Test 2. Multi-line comment
        code2 = "a = 1"
        test2 = ['/*', '*1*', '**2**', '*/', code2]
        exp2 = [code2]
        self.assertListEqual(exp2, self.su.remove_comments(test2), "failed test 2")
        exp3 = ["c = 3", "xyz"]
        test3 = ['c = 3 /*', '*1*', '**2**', '*/ xyz', ]
        self.assertListEqual(exp3, self.su.remove_comments(test3), "failed test 3")

    def test_is_variable(self):
        # Test 1. normal
        tests = ['fred', "abc_123", 'NoRmal']
        for test1 in tests:
            self.assertTrue(self.su.is_variable(test1), f"failed test 1 on string {test1}")
        # Test 2. not variable names
        tests = ["100", "2bad", "%symbol."]
        for test2 in tests:
            self.assertFalse(self.su.is_variable(test2), f"failed test 2 on string {test2}")

    def test_is_SAS_variable(self):
        tests = ['xyzzy', 'x1', '3no', '&xyzzy', '&xyzzy.', '3.4']
        expected = [True, True, False, False, True, False]
        for test, exp in zip(tests, expected):
            self.assertEqual(exp, self.su.is_SAS_variable(test), f'Expected {exp} for is_SAS_variable({test}')

    def test_extract_variables(self):
        # Test 1. normal.
        test1 = "min(a, b, c)"
        exp1 = ['min', 'a', 'b', 'c']
        self.assertListEqual(exp1, self.su.extract_variables(test1), f'failed test 1')
        # Test 2. normal.
        test2 = "v / w + x - y * z ** (1/2)"
        exp2 = ['v', 'w', 'x', 'y', 'z']
        self.assertListEqual(exp2, self.su.extract_variables(test2), f'failed test 2')

    def test_excel_col_to_int(self):
        # Test 1. normal.
        tests = {"A":1, "B": 2, "aa": 27, "C33": 3, "123": 0, " 12 d39": 4}
        for test, exp in tests.items():
            self.assertEqual(exp, self.su.excel_col_to_int(test), f'input {test} did not return {exp}')

    def test_digits_only(self):
        """
        Convert the digits-only part of the string like "AB14" to "14" and "D32" to "32".
        If there is more than one group of digits, just get the first group.
        From https://www.geeksforgeeks.org/python-extract-digits-from-given-string/
        :param my_string:
        :return:
        """
        # Test 1. normal.
        tests = {"A1": '1', "B23": '23', "noDigits": '', "C33": '33', " 123 ": '123', " 12 d 39": '1239'}
        for test, exp in tests.items():
            self.assertEqual(exp, self.su.digits_only(test), f'input {test} did not return {exp}')

    def test_digits_only_as_int(self):
        """
        Convert the digits-only part of the string like "AB14" to "14" and "D32" to "32".
        If there is more than one group of digits, just get the first group.
        From https://www.geeksforgeeks.org/python-extract-digits-from-given-string/
        :param my_string:
        :return:
        """
        # Test 1. normal.
        tests = {"A1": 1, "B23": 23, "noDigits": 0, "C33": 33, " 123 ": 123, " 12 d 39": 1239}

        for test, exp in tests.items():
            self.assertEqual(exp, self.su.digits_only_as_int(test), f'input {test} did not return {exp}')

    def test_parse_url(self):
        # Test 1. normal
        test1 = "https://www.sas.com/en_us/home.html"
        actual1 = self.su.parse_url(test1)
        self.assertEqual("https", actual1.scheme)
        self.assertEqual("www.sas.com", actual1.hostname)
        self.assertEqual("/en_us/home.html", actual1.path)
        # Test 2. The works
        test2 = "http://www.google.com?gws_rd=ssl#q=python"
        actual2 = self.su.parse_url(test2)
        self.assertEqual("http", actual2.scheme)
        self.assertEqual("gws_rd=ssl", actual2.query)
        self.assertEqual("q=python", actual2.fragment)

    def test_fill_string(self):
        # Test 1, centered
        my_str = 'xyzzy'
        str_len = len(my_str)
        exp1 = '$$$$' + my_str + '$$$$'
        act1 = self.su.fill_string(my_str, '$', fill_width=2*4+str_len)
        self.assertEqual(exp1, act1, 'Test 1 fail')
        # Test 2, left
        exp2 = my_str + '?'*10
        act2 = self.su.fill_string(my_str, '?', fill_width=10+str_len, alignment='left')
        self.assertEqual(exp2, act2, 'Test 2 fail')
        # Test 3, right
        exp3 = '@'*20 + my_str
        act3 = self.su.fill_string(my_str, '@', fill_width=20+str_len, alignment='right')
        self.assertEqual(exp2, act2, 'Test 3 fail')

    def test_convert_string_append_type(self):
        # Test 1, percent
        test1 = '99.75%'
        exp_val_1 = 0.9975
        exp_type_1 = 'Percent'
        act1 = self.su.convert_string_append_type(test1)
        self.assertEqual(exp_val_1, act1.value)
        self.assertEqual(exp_type_1, act1.cellType)
        # test 2, number with comma
        test2 = '102,309.58'
        exp_val_2 = 102309.58
        exp_type_2 = 'Comma'
        act2 = self.su.convert_string_append_type(test2)
        self.assertEqual(exp_val_2, act2.value)
        self.assertEqual(exp_type_2, act2.cellType)
        # test 3, plain string with a numbr
        test3 = 'December 3'
        exp_val_3 = test3
        exp_type_3 = 'Normal'
        act3 = self.su.convert_string_append_type(test3)
        self.assertEqual(exp_val_3, act3.value)
        self.assertEqual(exp_type_3, act3.cellType)


class TestLineAccmulator(unittest.TestCase):
    def setUp(self):
        logger.debug('Starting TestLineAccmulator')
        self.la = LineAccmulator()

    def test_add_line(self):
        first_line = 'hello, world'
        second_line = "How's it going?"
        self.la.add_line(first_line)
        self.la.add_line(second_line)
        exp = [first_line, second_line]
        self.assertListEqual(exp, self.la.contents)

    def test_add_lines(self):
        first_line = 'hello, world'
        second_line = "How's it going?"
        both_lines = [first_line, second_line]
        self.la.add_lines(both_lines)
        exp = [first_line, second_line]
        self.assertListEqual(exp, self.la.contents)

    def test_add_df(self):
        cu = CollectionUtil()
        exp = ['uno', 'dos', 'tres', 'quatro']
        d  = {'col1': exp}
        df = DataFrame(data=d)
        self.la.add_df(df)
        for el in exp:
            self.assertTrue(cu.any_string_contains(lines=self.la.contents, find_me=el))

    def test_contents_len(self):
        # Test 1, empty list
        self.assertEqual(0, self.la.contents_len())
        # Test 2, regular
        test2 = ['uno', 'dos', 'tres', 'quatro']
        self.la.add_lines(test2)
        self.assertEqual(len(test2), self.la.contents_len())



# Use the following to run standalone. In PyCharm, you try Run -> Unittests in test_StringUtil.py.
# if __name__ == '__main__':
#     unittest.main()