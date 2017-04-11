#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import model_from_json
import os
import pandas as pd
#from seg.seg_fingerprint import getActiveCropArea, getCenterCropArea
from Patches.makePatch import patchMaker
import numpy as np
import lvdUtil.liveUtil as lvd
from optparse import OptionParser
from pandas import Series
import time
import json
from keras.preprocessing import image
from scipy.misc import imread, imresize, imsave
import preModels.overSqueeze as ovs
import time

if __name__=="__main__":
    #%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    
    parser.add_option("-m","--modelHome", dest='mHome',
                  default="data/models/Patch/basic/2011/BiometrikaTrain/base32", help="Home directory of models")
    parser.add_option("-n", "--model", dest="mName",
                  default="basicP32_11_Bio_12-0.92.hdf5", help="model name")
    
    parser.add_option("-t", "--test", dest="testFolder", 
                      default="data/LivTest/LivDet2011/BiometrikaTest", help="Test Sensor Folder")
    parser.add_option("-b", "--binary", dest="binary", action='store_false', default=True, help="Binary Class?")
    parser.add_option("-r", "--result", dest="result",
                      default="data/results/Patch", help="result saving folder")
    parser.add_option("-p", "--preTrain", dest="preTrain", 
                      default="basic", help="Select pre-trained model")
    parser.add_option("-a", "--heatmap", dest="heatmap", action="store_true", 
                      default=False, help="Do not make heat map")
    parser.add_option("-s", "--savemap", dest="saveMap", action="store_true", 
                      default=False, help="Do not make heat map")
    parser.add_option('-l', '--savelocation', dest='saveLocation', default='data/heatmap/',
                      help='save location of heatmap image')
 
    options, args = parser.parse_args()

    modelHome = options.mHome
    modelFile = options.mName
    testFolder = options.testFolder
#    reFolder = options.result
    binary = options.binary
    preTrain = options.preTrain
    modelPath = os.path.expanduser(os.path.join('~', modelHome, modelFile))
    testPath = os.path.expanduser(os.path.join('~', testFolder))
    heatmap = options.heatmap
    saveMap = options.saveMap
    saveLocation = options.saveLocation
#    heatmap = True
#    saveMap = True

            
            

#%%
    label = lvd.loadLabelFromYear(testFolder.split("/")[2], testFolder.split("/")[3])
##
    if label is None:
        print("Check your data folder orders.")
        print("Check your test folder name. '/' is not needed at first" )
        os._exit(1)
##%% Model load
#    ## This part should be changed
#
    input_size = int(modelFile[modelFile.find("P")+1:modelFile.find("P")+3])
    reFolder = os.path.join(options.result, preTrain, str(input_size))
    model = ovs.convnet(preTrain, weights_path=modelPath, heatmap=heatmap,input_size=input_size)
    if saveMap:
        path1 = reduce(os.path.join, reFolder.split('/')[-2:]+testFolder.split('/')[-2:])
#        path2 = reduce(os.path.join, )
        heatFolder = os.path.expanduser(os.path.join("~",saveLocation, path1))
        
#        
    if heatmap:
        modelPrefix = modelFile[:3]+"H" 
    else:
        modelPrefix = modelFile[:3]
    extList = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
###
###       #%%
    failList = []
    cropStart = []
#    
    confusion = lvd.makeDataFrame(label, binary=binary)
    if binary:
        rLabel = lvd.binaryLabel(bg=True)
    else:
        rLabel = label
        
#############################################################################        
#############################################################################
#############################################################################
        
##%%
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
            
#                cropStart = []
            if heatmap:                
                reShapeImage = imgLoaded.reshape(1, imgLoaded.shape[0], imgLoaded.shape[1], 1)
                start_time = time.time() # Time checker
                features = model.predict(reShapeImage)
                counter.append(time.time() - start_time) # Time checker end
                predIndex = np.argmax(features, axis=3) # it looks like image (2D)
                unique, counts = np.unique(predIndex.ravel(), return_counts=True)
                if saveMap:
                    lastHeatFolder = os.path.join(heatFolder, os.path.basename(labelFolder))
                    if not os.path.isdir(lastHeatFolder):
                        os.makedirs(lastHeatFolder)
                    lastHeatImage = os.path.join(lastHeatFolder, img_path)
                    imsave(lastHeatImage, features[0,:,:,:])
#                histo = dict(zip(unique, counts))
#                histo.pop(rLabel["BG"], None) # remove BG label
                
            else:
                patches = patchMaker(imgLoaded, input_size)
    
                #%%
                batch = np.zeros((len(patches),)+(input_size, input_size, 1))
                for ind, pa in enumerate(patches):
                    batch[ind,:,:,0] = pa
                start_time = time.time() # Time checker
                features = model.predict(batch)
                counter.append(time.time() - start_time) # Time checker end
            # Add saving imagename function for incoorect result
                predIndex = np.argmax(features, axis=1)
                unique, counts = np.unique(predIndex, return_counts=True)

            histo = dict(zip(unique, counts))
            histo.pop(rLabel["BG"], None) # remove BG label
            
            inverse = [(value, key) for key, value in histo.items()]
            resLabel = max(inverse)[1]
            
            if rLabel[lName] != resLabel:
                failList.append(fullpath)
            
            confusion[lName][resLabel] += 1
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
##    fuck = confusion+aceFrame

    resultfolder = os.path.expanduser(os.path.join('~', reFolder))
    if not os.path.exists(resultfolder):
        os.makedirs(resultfolder)
    prefix = testFolder.split('/')[2][-2:]+'_'+testFolder.split('/')[3][:3]
    modelResult = modelPrefix+'_'+prefix+'_'+modelFile[:modelFile.rfind('.')]+'.xlsx'
    errResult = modelPrefix+prefix+'errors_'+modelFile[:modelFile.rfind('.')]+'.xlsx'
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