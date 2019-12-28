import unittest
import logging
from LogitUtil import logit
from StringUtil import StringUtil

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

    @logit()
    def test_find_first_substring_in_list(self):
        l = ['abc123', 'def234', 'abc456']
        self.assertIsNone(self.su.find_first_substring_in_list('notInList', l))
        self.assertEqual(l[0], self.su.find_first_substring_in_list('abc', l))
        self.assertEqual(l[2], self.su.find_first_substring_in_list('abc4', l))


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
        longer = '*' * 50
        my_max = 24
        truncated = '*' * my_max
        self.assertEqual(truncated, self.su.truncate(longer, max=my_max))

# Use the following to run standalone. In PyCharm, you try Run -> Unittests in test_StringUtil.py.
# if __name__ == '__main__':
#     unittest.main()