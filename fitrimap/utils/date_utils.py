# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:51:56 2025

@author: Labadmin
"""
from datetime import datetime, timedelta


def doy_to_month_day(year, doy):
    try:
        # January 1 of the given year
        start_date = datetime(int(year), 1, 1)

        # Add the DOY to the start date, subtracting 1 to account for January 1
        target_date = start_date + timedelta(days=float(doy) - 1)

        # Extract month and day
        month = str(target_date.month)
        day = str(target_date.day)

        if len(month) == 1:
            month = f'0{month}'
        if len(day) == 1:
            day = f'0{day}'

        return month, day
    except ValueError as e:
        raise ValueError(f"Invalid input: {e}. Ensure DOY is within the valid range for the year.")
