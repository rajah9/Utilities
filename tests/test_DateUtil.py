import logging
import unittest
from datetime import timedelta, datetime
import time
from LogitUtil import logit
from DateUtil import DateUtil

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


if __name__ == "__main__":
    unittest.main(argv=['first-arg-ignored'], exit=False)
