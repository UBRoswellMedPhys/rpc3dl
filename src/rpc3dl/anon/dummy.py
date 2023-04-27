# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 01:26:54 2023

@author: johna
"""

import random
import string

def dummy_data(size=8, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def dummy_CS():
    chars = string.ascii_uppercase + string.digits + '_'
    return dummy_data(chars=chars)

def dummy_string():
    chars = string.ascii_letters + string.digits
    return dummy_data(chars=chars)

def dummy_date():
    return '19000101'

def dummy_time():
    return '123000.00'

def dummy_datetime():
    return dummy_date() + dummy_time()

def dummy_age():
    return '099Y'

def dummy_int():
    return dummy_data(size=6,chars=string.digits)

def dummy_decimal():
    return dummy_data(size=5,chars=string.digits) + ".0"

def dummy_UID():
    UID = "1.2.246.352.221." + random.choice("123456789")
    while len(UID) < 59:
        UID += random.choice(string.digits)
    UID += random.choice("123456789")
    return UID

def dummy_patientID():
    return dummy_data(size=20)