import logging
import time
import unittest
from datetime import timedelta, datetime
from random import randint

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
        timestamp_float = time.mktime(testDate.timetuple())
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
    def test_asDate(self):
        expectedDate1 = datetime(2019, 4, 3)
        yyyy_mm_dd1 = self.du.asFormattedStr(expectedDate1)
        actual1 = self.du.asDate(yyyy_mm_dd1)
        self.assertEqual(expectedDate1, actual1)
        actual2 = self.du.asDate(expectedDate1)
        self.assertEqual(expectedDate1, actual2)
        # Test 3, timezone naive
        expected3 = self.du.now()
        actual3 = self.du.asDate(expected3, use_localize=False)
        self.assertEqual(expected3, actual3)
        # Test 4, timezone
        tz = 'US/Mountain'
        mst_du = DateUtil(tz)
        expected4 = mst_du.now()
        actual4 = mst_du.asDate(expected4, use_localize=True)
        self.assertEqual(expected4, actual4)

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

if __name__ == "__main__":
    unittest.main(argv=['first-arg-ignored'], exit=False)
