#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:31:02 2017

@author: park
"""
import os
### Patch Number check
numList = []
#listFolder =["BG  Body Double  Ecoflex  Fake  Live  Playdoh]
folder = "/home/cvlab/data/Patch/Train/P64/LivDet2015/GreenBit/manyClass"
ffolder = [os.path.join(folder, mat) for mat in os.listdir(folder)]
for fo in ffolder:
    num = len(os.listdir(fo))
    numList.append(num)
spFolder = folder.split("/")
print spFolder[6], spFolder[7], spFolder[8]
nameOfBase = [os.path.basename(bname) for bname in ffolder]
#print nameOfBase
#print numList
summation = 0
fakeNum = 0
for name, num in zip(nameOfBase, numList):
    if (name == "BG") or (name=="Fake") or (name=="Live"):
        print(name + " : ", num)
        if name=="Fake":
            fakeNum = num
    else:
        summation += num
print summation == fakeNum 



#LivDet2011 = "BiometrikaVal  DigitalVal  ItaldataVal  SagemVal"
#LivDet2013 = "BiometrikaVal  CrossMatchVal  ItaldataVal  SwipeVal"
#LivDet2015 = "CrossMatch  Digital_Persona  GreenBit  Hi_Scan"

LivDet2011 = "BiometrikaTrain  DigitalTrain  ItaldataTrain  SagemTrain"
LivDet2013 = "BiometrikaTrain  CrossMatchTrain  ItaldataTrain  SwipeTrain"
LivDet2015 = "CrossMatch  Digital_Persona  GreenBit  Hi_Scan"
