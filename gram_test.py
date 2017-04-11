#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:56:34 2017

@author: park
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import model_from_json
import os
import pandas as pd
from Patches.makePatch import patchMaker
import numpy as np
import lvdUtil.liveUtil as lvd
from optparse import OptionParser
from pandas import Series
import time
from keras.preprocessing import image
from scipy.misc import imread, imresize, imsave
import preModels.gramModel as gram
import time


if __name__=="__main__":
    #%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    
    parser.add_option("-m","--modelHome", dest='mHome',
                  default="data/models/Gram/2011/Bio", help="Home directory of models")
    parser.add_option("-n", "--model", dest="mName",
                  default="gss_Aug11_Bio_16-0.93.hdf5", help="model name")
    
    parser.add_option("-t", "--test", dest="testFolder", 
                      default="data/LivTest/LivDet2011/BiometrikaTest", help="Test Sensor Folder")
    parser.add_option("-b", "--binary", dest="binary", action='store_false', default=True, help="Binary Class?")
    parser.add_option("-r", "--result", dest="result",
                      default="data/results/Gram", help="result saving folder")

    options, args = parser.parse_args()

    modelHome = options.mHome
    modelFile = options.mName
    testFolder = options.testFolder
    binary = options.binary
    preTrain = modelFile[:3]
    modelPath = os.path.expanduser(os.path.join('~', modelHome, preTrain, modelFile))
    testPath = os.path.expanduser(os.path.join('~', testFolder))
    
#%%
    label = lvd.loadLabelFromYear(testFolder.split("/")[2], testFolder.split("/")[3])
###
    if label is None:
        print("Check your data folder orders.")
        print("Check your test folder name. '/' is not needed at first" )
        os._exit(1)
##%% Model load
##    ## This part should be changed
##
    reFolder = os.path.join(modelHome.replace("models", "results"), preTrain)
    input_shape = lvd.imageSizeLoader(testPath)
    if preTrain == 'gsl':
        model = gram.shallowGramModel(input_shape=input_shape, weights_path=modelPath)
    elif preTrain == 'gml':
        model = gram.get_MultipleGramModel(input_shape=input_shape, weights_path=modelPath)
    elif preTrain == 'gvs':
        model = gram.veryShallowGramModel(input_shape=input_shape, weights_path=modelPath)
    elif preTrain == 'gss':
        model = gram.get_smallGramModel(input_shape=input_shape, weights_path=modelPath)
    else:
        print("Can not find your model. check your preTrain option")
        os._exit(1) 
#    if preTrain == "gml":
#        model = gram.get_MultipleGramModel(input_shape = input_shape, weights_path=modelPath)
#    

    extList = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']

    failList = []

    confusion = lvd.makeDataFrame(label, binary=binary)
    if binary:
        rLabel = lvd.binaryLabel(bg=False)
    else:
        rLabel = label
###%%
    counter = []
    for lName, lNum in label.items():
        labelFolder = os.path.join(testPath,lName)
        ImageList = os.listdir(labelFolder)
        if not ImageList:
            print("%s has no files." % os.path.join(testPath,lName))
            os._exit(1) 
    #%%
#    ### for loop uncomment
        print("%s evaluation start." % lName)
        if binary:
            if lName == "Live":pass
            else:lName = "Fake"
#            
        for img_path in ImageList:
            ext = os.path.splitext(img_path)[1]
            if ext.lower() not in extList: # extension check
                continue
            fullpath = os.path.join(labelFolder, img_path)
            try:
                imgLoaded = imread(fullpath, flatten=True)*1/255. # Cross match is not operate on this
            except:
                print("%s is broken file." %img_path)
                continue
            
            reShapeImage = imgLoaded.reshape(1, imgLoaded.shape[0], imgLoaded.shape[1], 1)
            start_time = time.time() # Time checker
            features = model.predict(reShapeImage)
            counter.append(time.time() - start_time) # Time checker end

            predIndex = np.argmax(features) # it looks like image (2D)

            if rLabel[lName] != predIndex:
                failList.append(fullpath)
            
            confusion[lName][predIndex] += 1
        print("Done.")
#    
#
    far = confusion.ix[0][1]/np.sum(confusion['Fake'])
    frr = confusion['Live'][1]/np.sum(confusion['Live'])
    ace = (far+frr)/2*100  
    print ("Your ACE is : %.4f" % ace)
    confusion['ACE'] = Series(np.array([ace, 0]), index=confusion.index)
    avgTime = np.array(counter)[1:].mean()
    confusion['Time'] = Series(np.array([avgTime, 0]), index=confusion.index)

    resultfolder = os.path.expanduser(os.path.join('~', reFolder))
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    prefix = testFolder.split('/')[2][-2:]+'_'+testFolder.split('/')[3][:3]
    modelResult = modelFile[:modelFile.rfind('.')]+'.xlsx'
    errResult = 'errors_'+modelFile[:modelFile.rfind('.')]+'.xlsx'
    saveResultName = os.path.join(resultfolder, modelResult)
    writer = pd.ExcelWriter(saveResultName, engine='xlsxwriter')
    # Convert the dataframe to an XlsxWriter Excel object.
    confusion.to_excel(writer, sheet_name='Sheet1')
    writer.save()
#    
    saveErrorName = os.path.join(resultfolder, errResult)
    errwriter = pd.ExcelWriter(saveErrorName, engine='xlsxwriter')
    errFrame = pd.DataFrame(failList)
    errFrame.to_excel(errwriter, sheet_name='Sheet1')
    errwriter.save()