#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:18:46 2016

@author: park
"""

from lvdUtil.liveUtil import makeResultFigure
import os

sourceDir = "/home/park/data/SegM/models/sxrAnotAug"
exten = '.log'
fileList = os.listdir(sourceDir)
for name in fileList:
    if name.endswith('.log'):
        makeResultFigure(os.path.join(sourceDir, name))

