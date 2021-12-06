import logging
import unittest
from datetime import timedelta, datetime
from random import randint
from time import sleep, mktime

from DateUtil import DateUtil
from LogitUtil import logit

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Test_DateUtil(unittest.TestCase):
    def setUp(self):
        self.du = DateUtil()

    @logit()
    def test_now(self):
        tz_str = 'US/Pacific'
        test_du = DateUtil(tz=tz_str)
        expected1 = test_du.now()
        expected2 = self.du.now(tz_str)
        logger.debug(f'expected1: {test_du.asFormattedStr(expected1, "%Y-%m-%dT%H:%M:%S TZ %z")}')
        # The following test almost always comes to the same second, but will occasionally fail.
        self.assertEqual(test_du.asFormattedStr(expected1, "%Y-%m-%dT%H:%M:%S TZ %z"), test_du.asFormattedStr(expected2, "%Y-%m-%dT%H:%M:%S TZ %z"))

    @logit()
    def test_changeDay(self):
        testday = self.du.today()
        tomorrow = testday + timedelta(days=1)
        self.assertEqual(tomorrow, self.du.changeDay(testday, 1), 'tomorrow test fails')
        lastweek = testday - timedelta(days=7)
        self.assertEqual(lastweek, self.du.changeDay(testday, -7), 'last week test fails')

    def test_changeDate(self):
        start_yy = 2021
        start_mm = 6
        start_dd = 1
        start_date = self.du.intsToDateTime(myYYYY=start_yy, myMM=start_mm, myDD=start_dd) # Start on 6/1/21
        # Test 1, normal. Provide monthly changes from last month to Jan of next year.
        for delta in range(-1, 8):
            this_mm = start_mm + delta
            this_yy = start_yy
            if this_mm > 12:
                this_mm -= 12
                this_yy += 1
            exp_date = self.du.intsToDateTime(myYYYY=this_yy, myMM=this_mm, myDD=start_dd)
            self.assertEqual(exp_date, self.du.changeDate(start_date, 'months', delta))
        # Test 2, exception. Should return None if providing a timePeriod of 'eons'.
        self.assertIsNone(self.du.changeDate(start_date, 'eons', 1))

    @logit()
    def test_latestDay(self):
        thur = 3
        expectedLastThur = self.du.latestDay(thur)  # 0=Mo, 1=Tu, 2=We, 3=Th, ... 6=Su
        today = self.du.today()
        for i in range(0, -7, -1):
            testday = self.du.changeDay(today, i)
            logger.debug(f'About to test {self.du.asFormattedStr(testday)}, which is a {testday.weekday()}')
            if testday.weekday() == thur:
                break
        self.assertEqual(testday, expectedLastThur)

    @logit()
    def test_asFormattedStr(self):
        testDate = datetime(2018, 11, 20)
        yyyy_mm_dd = self.du.asFormattedStr(testDate)
        self.assertEqual('2018-11-20', yyyy_mm_dd)
        mmSlddSlyy = self.du.asFormattedStr(testDate, '%m/%d/%Y')
        self.assertEqual('11/20/2018', mmSlddSlyy)
        timestamp_float = mktime(testDate.timetuple())
        self.assertEqual(self.du.asFormattedStr(testDate), self.du.asFormattedStr(timestamp_float))

    @logit()
    def test_intsToDateTime(self):
        yy, mm, dd = 2018, 12, 19
        actual = self.du.intsToDateTime(yy, mm, dd)
        self.assertEqual(datetime(yy, mm, dd), actual)

    @logit()
    def test_as_es_daterange(self):
        d1 = self.du.as_es_daterange(days_back=1)
        expected_end_d1 = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(expected_end_d1.isoformat(), d1['lt'])

    @logit()
    def test_year_quarter_to_datetime(self):
        # Test 1, normal
        test1 = '2021q4'
        exp1 = datetime(2021, 10, 1)
        act1 = self.du.year_quarter_to_datetime(test1)
        self.assertEqual(act1, exp1, 'Failed test 1')

    @logit()
    def test_asDate(self):
        expectedDate1 = datetime(2019, 4, 3)
        yyyy_mm_dd1 = self.du.asFormattedStr(expectedDate1)
        actual1 = self.du.asDate(yyyy_mm_dd1)
        self.assertEqual(expectedDate1, actual1, 'Failed test 1')
        actual2 = self.du.asDate(expectedDate1)
        self.assertEqual(expectedDate1, actual2, 'Failed test 2')
        # Test 3, timezone naive
        expected3 = self.du.now()
        actual3 = self.du.asDate(expected3, use_localize=False)
        self.assertEqual(expected3, actual3, 'Failed test 3')
        # Test 4, timezone
        tz = 'US/Mountain'
        mst_du = DateUtil(tz)
        expected4 = mst_du.now()
        actual4 = mst_du.asDate(expected4, use_localize=True)
        self.assertEqual(expected4, actual4, 'Failed test 4')
        # Test 5, using a quarter format
        test5 = '2022Q1'
        expected5 = datetime(2022, 1, 1)
        actual5 = self.du.asDate(test5)
        self.assertEqual(expected5, actual5, 'Failed test 5')

    @logit()
    def test_today(self):
        actual = self.du.today()
        expected = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.assertEqual(actual, expected)

    @logit()
    def test_as_timestamp(self):
        first = self.du.as_timestamp()
        second = self.du.as_timestamp(first)
        self.assertEqual(first, second)

    @logit()
    def test_duration(self):
        now = self.du.now()
        exp_days = randint(1,50)
        exp_hours = randint(0,23)
        exp_minutes = randint(0,59)
        logger.debug(f'looking at time {exp_days} days, {exp_hours} hours, {exp_minutes} minutes in the future.')
        next = now + timedelta(days=exp_days, hours=exp_hours, minutes=exp_minutes)
        act_days, act_hours, act_minutes = self.du.duration(now, next)
        self.assertEqual(exp_days, act_days)
        self.assertEqual(exp_hours, act_hours)
        self.assertEqual(exp_minutes, act_minutes)

    @logit()
    def test_is_after(self):
        # Test 1, one day later.
        yy, mm, dd = 2021, 8, 25
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd)
        start_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD= dd - 1)
        self.assertTrue(self.du.is_after(the_date=the_date, start_date=start_date), 'Fail test 1')
        # Test 2, the same time (should return True)
        start_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_after(start_date, start_date), 'Fail test 2')
        # Test 3, the_date is slightly after (should return True)
        sleep(0.05)
        the_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_after(the_date, start_date), 'Fail test 3')

    @logit()
    def test_is_before(self):
        # Test 1, one day later.
        yy, mm, dd = 2021, 8, 25
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd)
        end_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd + 1)
        self.assertTrue(self.du.is_before(the_date=the_date, end_date=end_date), 'Fail test 1')
        # Test 2, the same time (should return True)
        the_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_before(the_date, the_date), 'Fail test 2')
        # Test 3, end_date is slightly after (should return True)
        from time import sleep
        sleep(0.05)
        end_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_before(the_date, end_date), 'Fail test 3')

    @logit()
    def test_is_equal(self):
        # Test 1, normal
        yy, mm, dd = 2021, 8, 25
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd)
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD= dd)
        self.assertTrue(self.du.is_equal(the_date=the_date, test_date=the_date), 'Fail test 1')
        # Test 2,
        start_date = self.du.now(tz_str='EST')
        sleep(0.05)
        the_date = self.du.now(tz_str='EST')
        self.assertFalse(self.du.is_equal(start_date, the_date), 'Fail test 2')

    @logit()
    def test_is_between(self):
        # Test 1, one day later. (Almost the same as test_is_after.)
        yy, mm, dd = 2021, 8, 25
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd)
        start_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD= dd - 1)
        self.assertTrue(self.du.is_between(the_date=the_date, start_date=start_date), 'Fail test 1')
        # Test 2, the same time (should return True)
        start_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_between(start_date, start_date), 'Fail test 2')
        # Test 3, the_date is slightly after (should return True)
        sleep(0.05)
        the_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_between(the_date, start_date), 'Fail test 3')
        # Test 4, one day later. (Almost the same as test_is_before.)
        start_date = None
        the_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd)
        end_date = self.du.intsToDateTime(myYYYY=yy, myMM=mm, myDD=dd + 1)
        self.assertTrue(self.du.is_between(the_date=the_date, end_date=end_date), 'Fail test 4')
        # Test 5, the same time (should return False)
        the_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_between(the_date=the_date, end_date=the_date), 'Fail test 5')
        # Test 6, end_date is slightly after (should return True)
        sleep(0.05)
        end_date = self.du.now(tz_str='EST')
        self.assertTrue(self.du.is_between(the_date, end_date=end_date), 'Fail test 6')
        # Test 7, both are none (and must return True)
        self.assertTrue(self.du.is_between(the_date, end_date=None, start_date=None), 'Fail test 7')
        # Test 8, using quarters
        test8 = '2021Q4'
        exp8 = datetime(2021, 10, 1)
        day_before = datetime(2021, 9, 30)
        day_after = datetime(2021, 10, 2)
        self.assertTrue(self.du.is_between(the_date=exp8, start_date=day_before, end_date=day_after), 'Fail test 8')


    @logit()
    def test_four_digit_year(self):
        # Test 1, normal.
        exp1 = 2015
        act1 = self.du.four_digit_year(exp1)
        self.assertEqual(exp1, act1, 'Fail test 1')
        # Test 2, two digits for the year. 0 means use the current year.
        test2 = 0
        exp2 = datetime.today().year
        act2 = self.du.four_digit_year(test2)
        self.assertEqual(exp2, act2, 'Fail test 2')
        # Test 3, regular two digits. 21 means 2015.
        test3 = 15
        exp3 = 2015
        act3 = self.du.four_digit_year(test3)
        self.assertEqual(exp3, act3, 'Fail test 3')
        # Test 4, None for year should return this year.
        test4 = None
        exp4 = datetime.today().year
        act4 = self.du.four_digit_year(test4)
        self.assertEqual(exp4, act4, 'Fail test 4')

    @logit()
    def test_first_of_month(self):
        # Test 1, normal.
        yyyy1, mm1 = 1981, 2
        exp1 = self.du.intsToDateTime(myYYYY=yyyy1, myMM=mm1, myDD=1)
        act1 = self.du.first_of_month(yyyy1, mm1)
        self.assertEqual(exp1, act1, 'Fail test 1')
        # Test 2, two-digit year.
        yyyy2, mm2 = 2021, 3
        yy2 = yyyy2 % 100
        exp2 = self.du.intsToDateTime(myYYYY=yyyy2, myMM=mm2, myDD=1)
        act2 = self.du.first_of_month(year=yy2, month=mm2)
        self.assertEqual(exp2, act2, 'Fail test 2')

    @logit()
    def test_first_of_quarter(self):
        # Test 1, normal
        yyyy1, qq1, mm1 = 1981, 2, 4 # first-of-quarter 2 should be April
        exp1 = self.du.intsToDateTime(myYYYY=yyyy1, myMM=mm1, myDD=1)
        act1 = self.du.first_of_quarter(year=yyyy1, quarter=qq1)
        self.assertEqual(exp1, act1, 'Fail test 1')
        # Test 2, normal
        yyyy2, qq2, mm2 = 2021, 4, 10 # first-of-quarter 4 should be Oct
        exp1 = self.du.intsToDateTime(myYYYY=yyyy2, myMM=mm2, myDD=1)
        act1 = self.du.first_of_quarter(year=yyyy2, quarter=qq2)
        self.assertEqual(exp1, act1, 'Fail test 2')

if __name__ == "__main__":
    unittest.main(argv=['first-arg-ignored'], exit=False)
