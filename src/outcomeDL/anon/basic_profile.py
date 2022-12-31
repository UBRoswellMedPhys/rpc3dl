# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:16:01 2022

@author: johna
"""

from pydicom.tag import Tag

"""
De-identification action per Table E.1-1 in DICOM standard

Table E.1-1a provides a legend for action codes

D - replace with non-zero length dummy value
Z - replace with zero length value
X - remove attribute
K - keep
C - clean identifying information but retain meaning
U - replace with a non-zero length UID that is internally consistent
Z/D - Z unless D is required to maintain IOD conformance (Type 2 vs Type 1)
X/Z - X unless Z is required to maintain IOD conformance (Type 3 vs Type 2)
X/D - X unless D is required to maintain IOD conformance (Type 3 vs Type 1)
X/Z/D - X unless Z or D is required to maintain IOD conformance
X/Z/U - X unless Z or U is required to maintain IOD conformance
"""
basic_profile = {
    Tag(('0008','0050')): 'Z',
    Tag(('0018','4000')): 'X',
    Tag(('0040','0555')): 'X/Z',
    Tag(('0008','0022')): 'X/Z',
    Tag(('0008','002A')): 'X/Z/D',
    Tag((''))
    }