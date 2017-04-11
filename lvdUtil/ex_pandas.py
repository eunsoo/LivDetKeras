#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:32:30 2016

@author: park
"""

import lvdUtil as lu
from pandas import Series, DataFrame
import pandas as pd

obj = Series([4, 7, -5, 3])

obj2 = Series([4, 7, -5, 3], index=['d','b','a','c'])

year = "LivDet2013"
sensor = "BiometrikaTrain"

label = lu.loadLabelFromYear(year, sensor)

slabel = Series(label)

labelSet = DataFrame(columns=[label.keys(), index=label.values()])
