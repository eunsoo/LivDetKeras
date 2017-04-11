# -*- coding: utf-8 -*-
#############################################################################
##### Run on the LivDetKeras folder #########################################
#############################################################################

#from seg.seg_fingerprint import weonjinSegmentation
##import util_livdet as util
import os
import cv2
from scipy.misc import imread
from optparse import OptionParser
import numpy as np
import random
import string
#
#
#################################################################################
#############   For making Patches ##############################################
#################################################################################

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
#    print(mInput.shape)
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

def imageLoader(path, extList, gray=False):
    ## Just for image loading (only for gray Fingerprint)
    ## It can be general image loader from a folder.
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
                # Load image
                imgLoaded = imread(fullpath, flatten=gray)
            except:
                print("%s is broken file." %img)
                continue
#            print(imgLoaded.shape)
            if imgLoaded is None : continue
            if len(imgLoaded.shape) < 2 : continue
            if len(imgLoaded.shape) == 4 : continue
            if not gray:
                if len(imgLoaded.shape) == 2: imgLoaded = np.tile(imgLoaded[:,:,None],3)        
                if imgLoaded.shape[2] == 4: imgLoaded = imgLoaded[:,:,:3]
                    
            imgs.append(imgLoaded)
            imgName.append(img)
    if(len(imgs) <= 0):
        print("There is no image file")
        return False
#    else:    
#        print("%d files are loaded." %len(imgs))
    return imgs, imgName
    
def patchMaker(img, stepSize):
    """
        make grid patches
    """
    hl = img.shape[0]/stepSize
    wl = img.shape[1]/stepSize
    hlist = np.array(range(hl))*stepSize
    wlist = np.array(range(wl))*stepSize
    patches = []    
    for h in hlist:
        for w in wlist:
            imgPatch =  img[h:h+stepSize,w:w+stepSize]
            patches.append(imgPatch)
        
    return patches

def makePatchData(img, stepSize, ratio, gray):
    """
    description : Make patches divided into background and foreground
    ---------------------------------------    
    Input :
        img - grayscale fingerprint image
        stepSize : Patch size
        ratio : ratio of segmented area per patch size
    Return :
        bg - background patch images
        fg - foreground patch images
        bgtl - background top left corners
        fgtl - foreground top left corners
    """
    seg = weonjinSegmentation(img)
    hl = img.shape[0]/stepSize
    wl = img.shape[1]/stepSize
    hlist = np.array(range(hl))*stepSize
    wlist = np.array(range(wl))*stepSize
    bg =[]
    fg = []
    bgtl = []
    fgtl = []
    if gray: 
        for i, h in enumerate(hlist):
            for j, w in enumerate(wlist):
                imgPatch =  img[h:h+stepSize,w:w+stepSize]
                segPatch = seg[h:h+stepSize,w:w+stepSize]
                rate = np.count_nonzero(segPatch)/float((stepSize*stepSize))
    #            result[i,j] = rate
                if rate > ratio:
                    fg.append(imgPatch[:,:,0])
                    fgtl.append((h, w))
                else:
                    bg.append(imgPatch[:,:,0])
                    bgtl.append((h,w))
    else:
        for i, h in enumerate(hlist):
            for j, w in enumerate(wlist):
                imgPatch =  img[h:h+stepSize,w:w+stepSize]
                segPatch = seg[h:h+stepSize,w:w+stepSize]
                rate = np.count_nonzero(segPatch)/float((stepSize*stepSize))
    #            result[i,j] = rate
                if rate > ratio:
                    fg.append(imgPatch)
                    fgtl.append((h, w))
                else:
                    bg.append(imgPatch)
                    bgtl.append((h,w))
        
    return bg, fg, bgtl, fgtl
    
def patchSave(patches, location, folder, png_params, prefix):
    """
        description : Save patches and make filelist
        -----------------------------------------------------
        Input :
            patches - patches
            location - top left corner of the patches
            folder - saving location
            png_params - compression level for png file
            
    """
    for bi, bb in enumerate(patches):
        hh = location[bi][0]
        ww = location[bi][1]
        unique = id_generator()
        iiName = prefix+'_'+unique+'_'+str(hh)+'x'+str(ww)+'.png'
        bNames = os.path.join(folder, iiName)
        cv2.imwrite(bNames, bb, png_params)
        
def id_generator(size=5, chars=string.ascii_uppercase+string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#################################################################################
#################################################################################

# "/media/park/hard/DBs/FAKE/lvedet/Old_Datasets/ValTrain/LivDet2011/Train/BiometrikaTrain/Materials"
# "/Dropbox/data/Patch/Train/LivDet2011/BiometrikaTrain/manyClass/Materials"


### This code only makes manyClass patches
if __name__ == "__main__":
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)
    parser.add_option("-s","--sfolder", dest='sFolder',
                  default="/media/park/hard/DBs/FAKE/lvedet/Old_Datasets/ValTrain/LivDet2011/Val/BiometrikaVal",
                  help="test result home")
    parser.add_option("-d", "--dfolder", dest="dFolder",
                  default="data/Patch", help="DB selection")
    parser.add_option("-c", "--color", action="store_false", dest="color", default = True, help="gray?")
    parser.add_option("-r", "--ratio", type="float", dest="ratio", default = 0.4, help="ratio of fg")
    parser.add_option("-b", "--bg", action="store_true", dest="saveBack", default = False, help="No BG save")
    options, args = parser.parse_args()

    sFolder = options.sFolder
    dFolder = options.dFolder
    gray = options.color
    ratio = options.ratio
    notBG_True = options.saveBack


    if os.path.isdir(sFolder):
        folderList = [os.path.join(sFolder,mat) for mat in os.listdir(sFolder)]             
    else:
        print("Can not find Source Folder")
        os._exit(1)

# Need to make for loop for Sensors folderList
    for sFolder in folderList:
#    sFolder = folderList[0]

        exts = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
        images, imgNames = imageLoader(sFolder, exts)
        if(images is False):
            print("There is no image in " + sFolder)
    #        continue
        
    # Need to make for loop for patchSize
    #    for stSize in [32, 48, 64]: # make below into for loop    
        for ra, stepSize in enumerate([32, 48, 64]):
#        stepSize = 32
            ra = ratio/(ra+1)
            pathList = os.path.join(sFolder.split("/"))
            destFolder = os.path.expanduser(os.path.join("~",dFolder, pathList[-3],
            "P"+str(stepSize),pathList[-4], pathList[-2], "manyClass", pathList[-1]))
        
            bgName = "BG"
            bgFolder = os.path.join(destFolder[:destFolder.rfind('/')], bgName)
            if not os.path.isdir(destFolder): os.makedirs(destFolder)
            if not os.path.isdir(bgFolder): os.makedirs(bgFolder)
            
            png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
            prefix = os.path.basename(destFolder)[:3]
            for ind, img in enumerate(images):
                prf = prefix+"_"+str(ind)
                bg, fg, bgtl, fgtl = makePatchData(img, stepSize, ra, gray)
                patchSave(fg, fgtl, destFolder, png_params, prf)
            if not notBG_True:
                patchSave(bg, bgtl, bgFolder, png_params, prf)
    print(pathList[-4]+", "+pathList[-2]+" is finished")
#        
#    print(pathList[-1]+" is finished")
###########################################################################
###########################################################################

    
    
#    destFolder = os.path.expanduser(os.path.join('~', dFolder, trVal, "P"+str(stepSize), subPath))
#    ssFolder = sFolder.replace("/Train/", "/"+trVal+"/")
#    if os.path.isdir(ssFolder)
##################################333

    # Background folder name
#    bgName = "BG"
#    bgFolder = os.path.join(destFolder[:destFolder.rfind('/')], bgName)
#    if os.path.isdir(sFolder):
#	    print ("Source Directory Selected.")
#	    # load images and their names in the folder
#	    exts = ['.bmp', '.png', '.gif', '.jpg', '.jpeg']
#	    images, imgNames = imageLoader(sFolder, exts, gray=gray)
#	    if(images is False):
#	        print("Bye~")
#	        raise SystemExit
#    else:
#        print ("Can not find source folder.")
#        raise SystemExit
#    
#    ## Making destination folders (Materials and BG)
#    if os.path.isdir(destFolder): print("Directory exists : " + destFolder)
#    else: os.makedirs(destFolder)
#    if os.path.isdir(bgFolder): print("BG folder exists.")
#    else: os.makedirs(bgFolder)
#    
#    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
#    prefix = os.path.basename(destFolder)[:3]
#    for ind, img in enumerate(images):
#        bg, fg, bgtl, fgtl = makePatchData(img, stepSize, ratio)
#        patchSave(fg, fgtl, destFolder, png_params, prefix)
#    if not notBG_True:
#        patchSave(bg, bgtl, bgFolder, png_params, prefix)
#        
#    print(pathList[-1]+" is finished")
        
    
    


    
        
        
# if __name__ == "__main__":

# ######### Patch Maker #####################################
# #==============================================================================
# # """
# # This one make patch data that has the same original folder architecture.
# # Uage : python makePatch.py 2011 
# # """
# 	use = "Usage : %prog [option]"
# 	parser = OptionParser(usage=use)

# 	parser.add_option("-y","--year", dest='year',
#                   default="LivDet2011", help="Year of Sensors")
# 	parser.add_option("-s","--step", dest='step',
#                   default=16, help="Step Size")
# 	options, args = parser.parse_args()

# 	my_livdet = "/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets"
# 	dataset = options.year
# 	rootDir = "/home/park/mnt/DBs/FAKE/lvedet/Old_Datasets/"+dataset
# 	trFolderName = "Training"
# 	patchName = "Patch"
# 	stepSize = int(options.step)
# 	rate = 0.70
# 	bgName = 'bg'
# 	fgName = 'fg'
# 	orFolderName = os.path.join(rootDir, trFolderName) 
# 	patchFolderName = os.path.join(rootDir, patchName, str(stepSize)+'x'+str(stepSize), trFolderName)
# #    direct = "C:\\Users\\park\\Dropbox\\100.Projects\\LivenessDetection\\python_scripts\\Data"
# 	#%% image and folder search pairs    
# 	direc, imgFiles = util.folderFiles(orFolderName, ['.bmp','.png'])
# 	#%% Make Destiantion derectory
# 	destFolder = util.makeDestFolder(patchFolderName, direc)
# 	#%% Make patches
# 	png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
# 	#%%
# 	for index, dt in enumerate(destFolder):
# 		backFolder = os.path.join(dt, bgName)
# 		foreFolder = os.path.join(dt, fgName)
# 		orgFolder = direc[index]
# 		util.makeBGFGFolder(backFolder)        
# 		util.makeBGFGFolder(foreFolder)        
		
# 		bgtxtName = util.makeTxtName(dt, bgName, backFolder)
# 		bfile = open(bgtxtName, 'w')

# 		fgtxtName = util.makeTxtName(dt, fgName, foreFolder)
# 		ffile = open(fgtxtName, 'w')        
		
# 		for ii, img in enumerate(imgFiles[index]):
# 			imgName = os.path.join(orgFolder, img)
# 			image = cv2.imread(imgName, 0)
# 			if image is None: continue
# 			else:
# 				segImg = seg.weonjinSegmentation(image)
# 				bg, fg, bgtl, fgtl = util.makePatchData(image, segImg, stepSize, rate)
# 				util.patchSave(bg, bgtl, backFolder, png_params, img, bfile)
# 				util.patchSave(fg, fgtl, foreFolder, png_params, img, ffile)
			
# #  patchSave function need to be changed. list of bfile is better. Saving to
# #  txt file is latter to finish making list of bfile.
# 	bfile.close()
# 	ffile.close()