"""
This script provides some date utilties.
"""
import logging
from datetime import timedelta, datetime
from LogitUtil import logit
from pytz import timezone
from typing import Union

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Class DateUtil last updated 26Oct19.
Interesting Python things:
* iso_format is a class-level variable that can be found with DateUtil.iso_format.
"""


class DateUtil:
    iso_format = "%Y-%m-%dT%H:%M:%S"
    ctime_format = "%a %b %d %H:%M:%S %Y"  # Parsing ctime like Thu Apr 11 10:36:18 2019
    def __init__(self, tz:str='US/Eastern'):

        # define eastern timezone
        self.my_tz = timezone(tz)

    @logit()
    def now(self, tz_str:str=None):
        tz = timezone(tz_str) if tz_str else self.my_tz
        logger.debug(f'now is using a timezone of {tz}')
        return datetime.now(tz)


    @logit(showArgs=False, showRetVal=True)
    def latestDay(self, dayOfWeek: int = 0) -> datetime:
        """
        Return the latest day (could be today) that corresponds to the given day of the week.
        We're using Python day conventions, using an integer, where Monday is 0 and Sunday is 6.
        """
        today = self.today()
        for i in range(7):
            testMe = today - timedelta(i)
            if testMe.weekday() == dayOfWeek:
                return testMe
        logger.error('Unable to find that day of week in the past 7 days (including today).')
        return None

    def asFormattedStr(self, myDate, myFormat: str = '%Y-%m-%d') -> str:
        """
        Convert myDate (which could be a datetime or float) to a formatted string, defaulting to the form 2018-11-20.
        Some formats I have used:

            %d	Day of the month as a zero-padded decimal number.	30
            %b	Month as locale’s abbreviated name.	Sep
            %B	Month as locale’s full name.	September
            %m	Month as a zero-padded decimal number.	09
            %y	Year without century as a zero-padded decimal number.	13
            %Y	Year with century as a decimal number.	2013
            %H	Hour (24-hour clock) as a zero-padded decimal number.	07
            %I	Hour (12-hour clock) as a zero-padded decimal number.	07
            %p	Locale’s equivalent of either AM or PM.	AM
            %M	Minute as a zero-padded decimal number.	06
            %S	Second as a zero-padded decimal number.	05

        For other formats, see https://strftime.org.
        """
        ans = None
        if type(myDate) == float:
            ans = datetime.utcfromtimestamp(myDate).strftime(myFormat)
        else:
            ans = myDate.strftime(myFormat)
        return ans

    def today(self):
        et_now = datetime.now(self.my_tz)
        return self.intsToDateTime(myYYYY=et_now.year, myMM=et_now.month, myDD=et_now.day)

    @logit(showArgs=False, showRetVal=True)
    def intsToDateTime(self, myYYYY: int, myMM: int, myDD: int, myHH: int = 0, myMin: int = 0,
                       mySec: int = 0) -> datetime:
        """
        Convert an integer year, month, day, hour, min, second to a Python dateTime.
        """
        ans = datetime(myYYYY, myMM, myDD, myHH, myMin, mySec)
        return ans

    @logit(showArgs=False, showRetVal=False)
    def asDate(self, date_or_str_or_timestamp:Union[str, float], myFormat: str = '%Y-%m-%d') -> datetime:
        """
        Convert a date or string or a timestamp (float) into a Python datetime.
        If it's a string or float, use myFormat to format it.
        """
        if type(date_or_str_or_timestamp) == str:
            return self.asDateTime(date_or_str_or_timestamp, myFormat)
        elif type(date_or_str_or_timestamp) == float:
            return self.timestamp_to_datetime(date_or_str_or_timestamp, myFormat)
        # it should already be a datetime.
        return date_or_str_or_timestamp

    @logit(showArgs=False, showRetVal=False)
    def changeDay(self, myDate: datetime, deltaDay: int = 1) -> datetime:
        """
        Given myDate, add (or subtract) n days to get n days in the future (past) depending on whether n is positive (or negative).
        """
        if deltaDay < 0:
            delta = 0 - deltaDay
            logger.debug('Subtracting {} days.'.format(delta))
        else:
            logger.debug('Adding {} days.'.format(deltaDay))
            pass

        return myDate + timedelta(days=deltaDay)

    @logit()
    def asDateTime(self, strToConvert: str, strFormat: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """
        Convert the string to a datetime.
        """
        ans = datetime.strptime(strToConvert, strFormat)
        return ans

    @logit()
    def timestamp_to_datetime(self, timestamp: float, strFormat: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        # Introducing some not-dangerous recursion.
        str = self.asFormattedStr(timestamp, strFormat)
        return self.asDateTime(str, strFormat)

    @logit()
    def build_es_daterange(self, start_date: datetime, end_date: datetime,
                           start_date_compare: str = "gte", end_date_compare: str = "lt") -> dict:
        """
        Take two dates and return a dictionary >= the start_date (usually a midnight time) and < end_date
        (usually a midnight time). If either date is None, then don't add it to the dictionary.
        """
        if start_date > end_date:
            raise ValueError(f'start_date {start_date} must precede end_date {end_date}.')

        ans = {}
        if start_date:
            ans[start_date_compare] = self.asFormattedStr(myDate=start_date, myFormat=self.iso_format)
        if end_date:
            ans[end_date_compare] = self.asFormattedStr(myDate=end_date, myFormat=self.iso_format)
        return ans

    def as_es_daterange(self, days_back: int = 1, start_date=None, end_date=None) -> dict:
        """
        Return an ElasticSearch date range.
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-range-query.html
        days_back  start_date   end_date    resulting start_date     resulting_end_date
        1          None         None        yesterday's midnight     this midnight
        None       None         None        yesterday's midnight     this midnight
        """
        ans = {}
        if not start_date and not end_date:
            # Pick the one 24 hour period days_back from now.
            end_date = self.today()
            delta = days_back if days_back < 0 else (0 - days_back)
            start_date = self.changeDay(end_date, deltaDay=delta)
            ans = self.build_es_daterange(start_date, end_date)

        return ans

    @logit(showArgs=False, showRetVal=False)
    def as_timestamp(self, dt=None):
        """
        Return the current datetime or float as a timestamp.
        Use the current date is dt is missing or None.
        """
        d = dt if dt else self.now()
        if type(d) == float:
            return d

        return d.timestamp()  # In the usual case when dt is a datetime. Works if Python >= 3.3


