# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:47:14 2016

@author: park
"""
import numpy as np
import cv2
from scipy.misc import imread
import os
import matplotlib.pyplot as plt
from optparse import OptionParser
np.random.seed(1337) # for reproducibility

def weonjinSegmentation(mInput):
    """
    description : segmentation of fingerprint image (binary)
    It's a Pyhon version of Weonjin Kim's segmentatin method
    ---------------------------------------    
    Input :
        mInput - gray scale ndarray
    Return :
        mTemp2 - segmented image (binary)
    """
    
    if len(mInput.shape)==3:
        mInput = mInput[:, :, 0]
    elif len(mInput.shape)>3:
        print ("Dimension is greater than 3.")
        return None
#    mInput = rgb2gray(mInput)
    coh = np.zeros(mInput.shape, dtype='f4')
    Gxx = np.zeros(mInput.shape, dtype='f4')
    Gyy = np.zeros(mInput.shape, dtype='f4')
    eisum = np.zeros(mInput.shape, dtype='f4')
    
    kernel1 = np.array([[-1],[1]])
    mOutput1 = cv2.filter2D(mInput, cv2.CV_8U, kernel1)
    mOutput2 = cv2.filter2D(mInput, cv2.CV_8U, kernel1.reshape(1,2))
    mOutput1 = mOutput1/255.
    mOutput2 = mOutput2/255.
    
    Gyy = mOutput1**2
    Gxx = mOutput2**2
    Gxy = mOutput1*mOutput2

    nWinSize = 5
    nWinHalf = nWinSize/2
#    h, w = 0, 0
    i = nWinHalf
    vcoh = []
    veisum = []
    while i < (mInput.shape[0]-nWinHalf):
        j = nWinHalf
        while j < (mInput.shape[1] - nWinHalf):
            mNumMatx = Gxx[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumx = mNumMatx.sum()
            mNumMaty = Gyy[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumy = mNumMaty.sum()
            mNumMatxy = Gxy[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            sumxy = mNumMatxy.sum()
            mNumMat = coh[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            eigenmax = np.abs((sumx + sumy + np.sqrt((sumx - sumy)**2 + 4 * sumxy*sumxy))) / 2.
            eigenmin = np.abs((sumx + sumy - np.sqrt((sumx - sumy)**2 + 4 * sumxy*sumxy))) / 2.
            eigensum=eigenmax+eigenmin
            eisum[i,j] = eigensum*2
            mNumMatsum = eisum[i-nWinHalf:i+nWinHalf+1, j-nWinHalf:j+nWinHalf+1]
            if(eigenmax+eigenmin == 0):
                coh[i,j] = 0
            else:
                coh[i,j] = (eigenmax-eigenmin)/(eigenmax+eigenmin)
            mNumMat[:] = coh[i,j]
            mNumMatsum[:] = eigensum*2
            vcoh.append(coh[i,j])
            veisum.append(eisum[i,j])
            j = j+nWinSize
        i = i+ nWinSize
    mcoh = np.array(vcoh)
    meisum = np.array(veisum)
    mcoh *= 255
    meisum *= 255
    mcoh = mcoh.astype(np.uint8)
    meisum = meisum.astype(np.uint8)
    thcoh = cv2.threshold(mcoh, 0, 255, cv2.THRESH_OTSU)[0]
    theisum = cv2.threshold(meisum, 0, 255, cv2.THRESH_OTSU)[0]
    mTemp = np.copy(coh)
    mTemp2 = np.copy(eisum)
    #%%
    mask2 = (mTemp>(thcoh*2.)/(255*3)) & (mTemp2>(theisum/(255.*2)))
    mask1 = (mTemp>(thcoh)/(255.*2)) & (mTemp2>((theisum*2)/(255.*3)))
    mask3 = ~(mask2 | mask1)
    mTemp2[mask2] = 255
    mTemp2[mask1] = 255
    #%%
    mTemp2 [mask3] = 0
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel5 =cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    
    mTemp2 = cv2.dilate(mTemp2, kernel5, iterations = 1)
    mTemp2 = cv2.morphologyEx(mTemp2, cv2.MORPH_CLOSE, kernel4,iterations = 2)
    mTemp2 = cv2.morphologyEx(mTemp2, cv2.MORPH_OPEN, kernel3,iterations = 10)
    return mTemp2
    
def imageFolderLoader(path, extList, cropSize):
    ImageList = os.listdir(path)
    imgs = []
    imgName = []
    if not ImageList:
        print("%s has no files." % path)
        return False             
    else:
        for img in ImageList:
            ext = os.path.splitext(img)[1]
            if ext.lower() not in extList:
                continue
            fullpath = os.path.join(path, img)
            try:
                imgLoaded = imread(fullpath)
            except:
                print("%s is broken file." %img)
                continue
            if imgLoaded is None : continue
            if len(imgLoaded.shape) < 2 : continue
            if len(imgLoaded.shape) == 4 : continue
            if len(imgLoaded.shape) == 2: imgLoaded = np.tile(imgLoaded[:,:,None],3)
            if imgLoaded.shape[2] == 4: imgLoaded = imgLoaded[:,:,:3]
            if imgLoaded.shape[2] > 4: continue    
            if(min(imgLoaded.shape[0], imgLoaded.shape[1])<cropSize):
                continue
            imgs.append(imgLoaded)
            imgName.append(img)
    if(len(imgs) <= 0):
        print("There is no image file")
        return False
    else:    
        print("%d files are loaded." %len(imgs))
    return imgs, imgName

def sizeFitChecker(start, end, Crop):
    startCheck = all(start < 0)
    if startCheck:
        print("Start index must be positive.")
    ends = start+Crop
    endCheck = all(ends > end)
    if endCheck:
        print("End index must be smaller than image.")
    return not (startCheck or endCheck)

def getCenterCropArea(image, cropSize=224):
    rows, cols = image.shape[0], image.shape[1]
    if (rows < cropSize) or (cols < cropSize):
        print("Image size is not fit.")
        return False
     # image center crop
    hCrop = cropSize/2
    rowStart= rows/2-hCrop
    colStart = cols/2-hCrop
    if sizeFitChecker(np.array([rowStart, colStart]),np.array([rows,cols]), cropSize):
        return [rowStart, colStart]
    else:
        print("Center of patch is not fit into image")
        return False
    
    
def getActiveCropArea(image, cropSize=224, numRcrop=4, centerOnly=False):
    cropActive = []
    # Assume one image    
    rows, cols = image.shape[0], image.shape[1]
    if (rows < cropSize) or (cols < cropSize):
        print("Image size is not fit.")
        return False 
    hCrop = cropSize/2
    segImg = weonjinSegmentation(image)
    if segImg is not None:    
        nonZeros = np.nonzero(segImg)
        if len(nonZeros[0]) < 1:
            print ("Segmentation fail")
            return False
    else:
        print("Segmentation fail beuase dimension mismatch.")
    # Active area center crop
    realRowCenter = int(np.mean(nonZeros[0]))
    realColCenter = int(np.mean(nonZeros[1]))
    # Row center matching 
    if (realRowCenter - hCrop) < 0:
        realRowCenter = realRowCenter+(hCrop-realRowCenter)
    elif (realRowCenter + hCrop) > rows :
        realRowCenter = realRowCenter-(realRowCenter+hCrop-rows)
    # Col neter matching
    if (realColCenter - hCrop) < 0:
        realColCenter = realColCenter+(hCrop-realColCenter)
    elif (realColCenter + hCrop) > cols :
        realColCenter = realColCenter-(realColCenter+hCrop-cols)
    rowStart = realRowCenter - hCrop
    colStart = realColCenter - hCrop
    if sizeFitChecker(np.array([rowStart, colStart]),np.array([rows,cols]), cropSize):
        cropActive.append([rowStart, colStart])
    else:
        print("Active center crop size is not fit to image.")
    
    if centerOnly == True:
        return cropActive
        
    # Random active area Crop
    indexRow= np.logical_and((nonZeros[0]-hCrop) >= 0, (nonZeros[0]+hCrop) <= rows)
    indexCol = np.logical_and((nonZeros[1]-hCrop) >= 0, (nonZeros[1]+hCrop) <= cols)
    eIndex = np.logical_and(indexRow, indexCol)
    activeArea = [nonZeros[0][eIndex], nonZeros[1][eIndex]]
    numActive = len(activeArea[0])
    selected = np.random.randint(numActive, size=numRcrop)
    for sel in selected:
        rowStart = activeArea[0][sel] - hCrop
        colStart = activeArea[1][sel] - hCrop
        if sizeFitChecker(np.array([rowStart, colStart]),np.array([rows,cols]), cropSize):
            cropActive.append([rowStart, colStart])
    return cropActive


def saveRandomImages(destFolder, images, imgNames, cropSize=224):
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    for img, iName in zip(images, imgNames):
        cropStart = []
        point = getCenterCropArea(img, cropSize)
        if point:
            cropStart.append(point)
        else:
            print("Image is so so small")
            continue
        points = getActiveCropArea(img, cropSize=cropSize, numRcrop=4)
        if points is not False:
            if (len(points)>0):
                cropStart = cropStart + points
            else:
                print("There is no active active area")
                continue
        for i, crop in enumerate(cropStart):
            sRow, sCol = crop[0], crop[1]
            cropped = img[sRow:sRow+cropSize, sCol:sCol+cropSize, :]
            oName = iName[:iName.rfind('.')]+'_'+str(i)+'.png'
            saveName = os.path.join(destFolder, oName)
            try:
                cv2.imwrite(saveName, cropped, png_params)
            except:
                print("Patche save is fail : %s" % saveName)
                continue
    print("%s saved" % destFolder)

def saveActiveImages(destFolder, images, imgNames, cropSize=224, centerOnly=False):
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    for img, iName in zip(images, imgNames):
        points = getActiveCropArea(img, cropSize=cropSize, numRcrop=4, centerOnly=centerOnly)
        if points is False:
            print("There is no active active area")
            continue
        else:
            crop = points[0]
        sRow, sCol = crop[0], crop[1]
        cropped = img[sRow:sRow+cropSize, sCol:sCol+cropSize, :]
        oName = iName[:iName.rfind('.')]+'_'+'0'+'.png'
        saveName = os.path.join(destFolder, oName)
        try:
            cv2.imwrite(saveName, cropped, png_params)
        except:
            print("Patche save is fail : %s" % saveName)
            continue
    print("%s saved" % destFolder)


    
if __name__=="__main__":
    #%% make segmented blocks
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)

    parser.add_option("-s","--sfolder", dest='sFolder',
                  default="/home/park/data/LivTest/LivDet2011/BiometrikaTest/Live",
                  help="test result home")
    parser.add_option("-d", "--dfolder", dest="dFolder",
                  default="/home/park/data/NewLivTest/LivDet2011/BiometrikaTest", help="DB selection")
    parser.add_option("-c", "--crops", dest="crop",
                  default=224, help="CropSize") 
    options, args = parser.parse_args()
    
    sFolder = options.sFolder
    dFolder = options.dFolder
    cropSize = int(options.crop)
    if os.path.isdir(sFolder):
        print ("Source Directory Selected.")
        # load images and their names in the folder
        exts = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
        images, imgNames = imageFolderLoader(sFolder, exts, cropSize)
        if(images is False):
            print("Bye~")
            raise SystemExit
    else:
        print ("Can not find source folder.")
        raise SystemExit
        
    destFolder = os.path.join(dFolder, os.path.basename(sFolder))    
    if os.path.isdir(destFolder): print("Directory exists : " + destFolder)
    else:
        os.makedirs(destFolder)
        print("Make Dir : " + destFolder)
#%%
    # saveRandomImages(destFolder, images, imgNames, cropSize=cropSize)
    saveActiveImages(destFolder, images, imgNames, cropSize=cropSize, centerOnly=True)
    print ("============= Finished ===============")

#        
#    
    
    
    
    

    