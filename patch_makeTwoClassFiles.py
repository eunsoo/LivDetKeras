#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:38:25 2017

@author: park
"""

import os
from optparse import OptionParser
import shutil
import lvdUtil.liveUtil as lvd

use = "Usage : %prog [option]"
parser = OptionParser(usage=use)

# Have to add expand user
patchPath = "/home/park/data/Patches"
paSizeFolder = ["P32", "P48", "P64"]
yearFolder = ["LivDet2011", "LivDet2013", "LivDet2015"]

training = "Training"
manyClass = "manyClass"
twoClass = "twoClass"
fake = "Fake"

# many Class and destination access
for paSize in paSizeFolder:
    for year in yearFolder:
        rootPath = os.path.join(patchPath, paSize, year, training)
        sensorName = [(os.path.join(rootPath, seName, manyClass), os.path.join(rootPath, seName, twoClass)) for seName in os.listdir(rootPath)]
        finalFolder = []
        for source, dest in sensorName:
            for matName in os.listdir(source):
                if matName == "Live":
                    finalFolder.append((os.path.join(source, matName), os.path.join(dest, matName)))
                elif matName =='BG':
                    finalFolder.append((os.path.join(source, matName), os.path.join(dest, matName)))
                else:
                    finalFolder.append((os.path.join(source, matName), os.path.join(dest, fake)))
        
        
        #    finalFolder = finalFolder+[(os.path.join(source, matName), os.path.join(dest, matName)) for matName in os.listdir(source)]
                               
        for sourceD, destD in finalFolder:
            if os.path.isdir(destD) : print("Directory exists : " + destD)
            else:
                print("Make Directory : " + destD)
                os.makedirs(destD)
            for imgName in os.listdir(sourceD):
                shutil.copy(os.path.join(sourceD, imgName), destD)

# Two class access
#
#for seName in sensorName:
#    os.path.join(rootPath, seName, manyClass)

