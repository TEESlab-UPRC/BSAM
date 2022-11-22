"""
This module is a library of custom functions used in the wmsim application
"""

import os
import datetime


class cd:
    """
    Context manager for changing the current working python directory
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def check_datetime_equality_month(date_1, date_2):
    """
    Checks datetime equality (month only)
    """
    if date_1.year == date_2.year and date_1.month == date_2.month:
        return True
    else:
        return False


def check_datetime_equality_day(date_1, date_2):
    """
    Checks datetime equality (day only)
    """
    if date_1.year == date_2.year \
        and date_1.month == date_2.month \
        and date_1.day == date_2.day:
            return True
    else:
        return False


def shift_months(date,months):
    """
    Shifts the date by the amount of months specified, not changing the exact current day
    TODO: Fix a case when shifting to 29th of Feb
    """
    years_to_shift = abs(int(months/12))
    months_to_shift = abs(months) % 12

    # if we are adding months
    if months > 0:
        if date.month + months_to_shift > 12:
            new_year = date.year + years_to_shift + 1
            new_month = date.month + months_to_shift - 12
        else:
            new_year = date.year + years_to_shift
            new_month = date.month + months_to_shift

    # if we are subtracting months
    else:
        if date.month - months_to_shift < 1:
            new_year = date.year - years_to_shift-1
            new_month = 12 - (months_to_shift - date.month)
        else:
            new_year = date.year - years_to_shift
            new_month = date.month - months_to_shift

    new_date = datetime.datetime(new_year,new_month,date.day)
    return new_date

