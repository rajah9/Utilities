"""
This script provides some date utilities.
"""
import logging
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from LogitUtil import logit
from pytz import timezone
from typing import Union, Tuple

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Class DateUtil last updated 26Oct19.
Interesting Python things:
* iso_format is a class-level variable that can be found with DateUtil.iso_format.
* Uses Tuple to return three values
"""


class DateUtil:
    iso_format = "%Y-%m-%dT%H:%M:%S"
    ctime_format = "%a %b %d %H:%M:%S %Y"  # Parsing ctime like Thu Apr 11 10:36:18 2019
    def __init__(self, tz:str='US/Eastern'):

        # define eastern timezone
        self.my_tz = timezone(tz)

    @logit()
    def now(self, tz_str:str=None) -> datetime:
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
            %a  Day of week. Mon
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

    def intsToDateTime(self, myYYYY: int, myMM: int, myDD: int, myHH: int = 0, myMin: int = 0,
                       mySec: int = 0) -> datetime:
        """
        Convert an integer year, month, day, hour, min, second to a Python dateTime.
        """
        ans = datetime(myYYYY, myMM, myDD, myHH, myMin, mySec)
        return ans

    @logit(showArgs=False, showRetVal=False)
    def asDate(self, date_or_str_or_timestamp:Union[str, float], myFormat: str = '%Y-%m-%d', use_localize:bool=False) -> datetime:
        """
        Convert a date or string or a timestamp (float) into a Python datetime.
        If it's a string or float, use myFormat to format it.
        """
        if type(date_or_str_or_timestamp) == str:
            return self.asDateTime(date_or_str_or_timestamp, myFormat, use_localize)
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

    def changeDate(self, myDate: datetime, timePeriod: str = 'months', delta: int = 1) -> datetime:
        """
        Return a date in the future or past, incremented by the given timePeriod.
        For more information on timePeriod, see https://dateutil.readthedocs.io/en/stable/relativedelta.html.
        :param myDate: datetime, like "Jan 1, 2021"
        :param timePeriod: string that relativeDelta understands, such as 'days', 'weeks', 'months', 'years', for example.
        :param delta: how many timePeriods in the future (if > 0)  or past (if < 0), example: 12
        :return: a datetime in the future or past, for example, "Jan 12, 2022"
        """
        relative_dict = {timePeriod: delta} # This is usually like {'months': 1}
        try:
            ans = myDate + relativedelta(**relative_dict)
        except TypeError as e:
            logger.warning(f'Got exception: {e}')
            logger.warning(f'Cannot handle a relative delta with a timeperiod of {timePeriod} and a delta of {delta}. Returning None.')
            return None
        return ans

    @logit()
    def asDateTime(self, strToConvert: str, strFormat: str = "%Y-%m-%d %H:%M:%S", use_localize: bool = False) -> datetime:
        """
        Convert the string to a datetime.
        """
        ans = datetime.strptime(strToConvert, strFormat)
        if use_localize:
            return self.my_tz.localize(ans).astimezone(self.my_tz)
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


    def duration(self, start_date: datetime, end_date: datetime) -> Tuple[int, int, int]:
        """
        Return the days, hours, and minutes between these dates.
        Taken from https://stackoverflow.com/questions/1345827/how-do-i-find-the-time-difference-between-two-datetime-objects-in-python
        :param start_date:
        :param end_date: should be later than start_date
        :return: days, hours, minutes between dates.
        """
        duration = end_date - start_date

        def days():
            return duration.days
        def hours():
            return int(duration.seconds / 3600) # 3600 seconds in an hour
        def minutes():
            hrs = hours()
            return int((duration.seconds - 3600 * hrs) / 60) # 60 seconds in a minute

        return days(), hours(), minutes()

    def is_after(self, the_date: datetime, start_date: datetime) -> bool:
        """
        return True if the_date is after start_date (by at least 1 ms);
        :param the_date: Date under examination.
        :param start_date: start date to compare the_date to
        :return: the_date > start_date
        """
        ans = the_date > start_date
        return ans

    def is_before(self, the_date: datetime, end_date: datetime) -> bool:
        """
        Return True if the_date is before end_date.
        :param the_date: Date under examination.
        :param end_date: end date to compare the_date to
        :return:
        """
        return the_date < end_date


    def is_between(self, the_date: datetime, start_date: datetime = None, end_date: datetime = None) -> bool:
        """
        Return True if start_date < the_date < end_date (strict inequality).
        If start_date is None, use is_before.
        If end_date is None, use is_after.
        If both start_date and end_date are None, return True.
        Raise an error in start_date > end_date.
        :param the_date:
        :param start_date: earlier than end_date
        :param end_date: later than start_date
        :return:
        """
        if start_date and end_date:
            if start_date < end_date:
                return self.is_before(the_date=the_date, end_date=end_date) and self.is_after(the_date=the_date, start_date=start_date)
            else:
                raise ValueError(f'start_date must precede end_date')
        elif start_date:
            return self.is_after(the_date=the_date, start_date=start_date)
        elif end_date:
            return self.is_before(the_date=the_date, end_date=end_date)
        else:
            return True
