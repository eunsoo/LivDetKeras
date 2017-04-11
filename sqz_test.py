#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 06:03:45 2016

@author: park
"""
#%%
#from keras.models import load_model, model_from_json
from keras.models import model_from_json
import os
import pandas as pd
from seg.seg_fingerprint import getActiveCropArea, getCenterCropArea
import numpy as np
import lvdUtil.liveUtil as lvd
from optparse import OptionParser
from pandas import Series
import time
import json
from keras.preprocessing import image
from scipy.misc import imread, imresize

#%%
#%%
if __name__=="__main__":
    #%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    
    parser.add_option("-m","--modelHome", dest='mHome',
                  default="data/SegM/models/Cross/sb", help="Home directory of models")
    parser.add_option("-n", "--model", dest="mName",
                  default="sb13_Cro_twoClass_11240104.hdf5", help="model name")
    
    parser.add_option("-t", "--test", dest="testFolder", 
                      default="data/LivTest/LivDet2013/CrossMatchTest/", help="Test Sensor Folder")
    parser.add_option("-b", "--binary", dest="binary", action='store_false', default=True, help="Binary Class?")
    parser.add_option("-r", "--result", dest="result",
                      default="data/results/sb", help="result saving folder")
    parser.add_option("-a", "--active", dest="active", action="store_true",
                      default=False, help="activeArea only?")

    options, args = parser.parse_args()
    
    modelHome = options.mHome
    modelFile = options.mName
    testFolder = options.testFolder
    reFolder = options.result
    binary = options.binary
    active = options.active
    
    modelPath = os.path.expanduser(os.path.join('~', modelHome, modelFile))
    testPath = os.path.expanduser(os.path.join('~', testFolder))
#%%
    label = lvd.loadLabelFromYear(testFolder.split("/")[2], testFolder.split("/")[3])
    if label is None:
        print("Check your data folder orders.")
        print("Check your test folder name. '/' is not needed at first" )
        raise
    #%% Model load
    ## This part should be changed
    stringOb = open(modelPath[:modelPath.rfind('_')]+'.json').read()
    modelconf = json.loads(stringOb)
    model = model_from_json(modelconf)
    model.load_weights(modelPath)
    
    cropSize = 224
    modelPrefix = modelFile[:3]

    if active:
        print("Test on Active area is selected.")

    if modelPrefix == 'inc':
        input_shape = (299,299,3)
    else:
        input_shape = (224,224,3)
    #%%
    extList = exts = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']

    #%%
    failList = []
    cropStart = []
    
    confusion = lvd.makeDataFrame(label, binary=binary)
    if binary:
        rLabel = lvd.binaryLabel()
    else:
        rLabel = label
    #########################################    
    #%%
## for loop uncomment  
    for lName, lNum in label.items():
        labelFolder = os.path.join(testPath,lName)
        ImageList = os.listdir(labelFolder)
        if not ImageList:
            print("%s has no files." % os.path.join(testPath,lName))
            raise ValueError  
    #%%
    ### for loop uncomment
        print("%s evaluation start." % lName)
        if binary:
            if lName == "Live":pass
            else:lName = "Fake"
    
        for img_path in ImageList:
            ext = os.path.splitext(img_path)[1]
            if ext.lower() not in extList: # extension check
                continue
            fullpath = os.path.join(labelFolder, img_path)
            try:
                # At first I need to test a few example along the models
                # This two line always works
#                imgLoaded = image.load_img(fullpath) # real case is scipy imread
#                imgLoaded = image.img_to_array(imgLoaded).astype('uint8')
                imgLoaded = imread(fullpath) # Cross match is not operate on this
            except:
                print("%s is broken file." %img_path)
                continue
            if imgLoaded is None : continue
            if len(imgLoaded.shape) < 2 : continue
            if len(imgLoaded.shape) == 4 : continue
            if len(imgLoaded.shape) == 2: imgLoaded = np.tile(imgLoaded[:,:,None],3)
            if imgLoaded.shape[2] == 4: imgLoaded = imgLoaded[:,:,:3]
            if imgLoaded.shape[2] > 4: continue    
            if(min(imgLoaded.shape[0], imgLoaded.shape[1])<cropSize):
                continue
            #%%
#            if len(imgLoaded.shape) == 2: imgLoaded = np.tile(imgLoaded[:,:,None],3) 
            if active:
                cropStart = []
                points = getActiveCropArea(imgLoaded, cropSize=cropSize, centerOnly=True)
                if points is not False:
                    if (len(points)>0):
                        cropStart = cropStart + points
                    else:
                        print("There is no active active area")
                        continue
            else:
                cropStart = []
                point = getCenterCropArea(imgLoaded, cropSize)
                if point:
                    cropStart.append(point)
                else:
                    print("Image is so small")
    #                imgNotLoaded.append(img_path)
                    continue
                points = getActiveCropArea(imgLoaded, cropSize=cropSize, numRcrop=4)
                if points is not False:
                    if (len(points)>0):
                        cropStart = cropStart + points
                    else:
                        print("There is no active active area")
                        continue
            #%%
            batch = np.zeros((len(cropStart),)+input_shape)
            #%%
            for i,crop in enumerate(cropStart):
                sRow, sCol = crop[0], crop[1]
#                img = image.load_img(fullpath)
#                x = image.img_to_array(img)
#                cropped = x[sRow:sRow+cropSize, sCol:sCol+cropSize, :]
                cropped = imgLoaded[sRow:sRow+cropSize, sCol:sCol+cropSize, :]
                if modelPrefix == 'inc':
                    cropped = imresize(cropped, [input_shape[0], input_shape[1]])
                else:
                    pass
#                x = image.img_to_array(cropped)
                batch[i,:] = cropped*1./255
            #    batch = batch*1./255
            #%%    
            #########################################
            
            features = model.predict(batch)
            # Add saving imagename function for incoorect result
            predIndex = np.argmax(np.mean(features, axis=0))
            if rLabel[lName] != predIndex:
                failList.append([fullpath, cropStart, features])
            
            confusion[lName][predIndex] += 1
        print("Done.")
    

    far = confusion.ix[0][1]/np.sum(confusion['Fake'])
    frr = confusion['Live'][1]/np.sum(confusion['Live'])
    ace = (far+frr)/2*100  
    print ("Your ACE is : %.4f" % ace)
    confusion['ACE'] = Series(np.array([ace, 0]), index=confusion.index)
#    fuck = confusion+aceFrame

    resultfolder = os.path.expanduser(os.path.join('~', reFolder))
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    prefix = testFolder.split('/')[2][-2:]+'_'+testFolder.split('/')[3][:3]
    modelResult = prefix+'_'+modelFile[:modelFile.rfind('.')]+'.xlsx'
    errResult = prefix+'errors_'+modelFile[:modelFile.rfind('.')]+'.xlsx'
    saveResultName = os.path.join(resultfolder, modelResult)
    writer = pd.ExcelWriter(saveResultName, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    confusion.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    
    saveErrorName = os.path.join(resultfolder, errResult)
    errwriter = pd.ExcelWriter(saveErrorName, engine='xlsxwriter')
    errFrame = pd.DataFrame(failList)
    errFrame.to_excel(errwriter, sheet_name='Sheet1')
    errwriter.save()
#    with open(errResult, 'w') as errFile:
#        for item in failList:
#            errFile.writelines?
        