# -*- coding: utf-8 -*-


#from keras.preprocessing import image

"""
    This script is for squeezeNet having pretrained weight.
    In this code, only final convolution layer is trained first.
    After that, all layers are also trained along a few epoch.
"""
from keras.optimizers import Nadam, Adamax
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator
import os
import preModels.gramModel as gram

import lvdUtil.liveUtil as lvd
#import json

#%%
    
if __name__ == '__main__':
#%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)

    parser.add_option("-d","--dataHome", dest='dataHome',
                  default="data", help="Home directory of data")
    parser.add_option("-s", "--sensor", dest="sensor",
                  default="ValTrain/LivDet2011/Train/BiometrikaTrain", help="Sensor folder")
    parser.add_option("-p", "--pretrain", dest="preTrain", default="gml", help="Pretraining Model")
    parser.add_option("-r", "--resultFolder", dest="result",
                      default="models/Gram/2011/Bio/gml", help="saving folder")
    parser.add_option("-e", "--epoch", dest="nb_epoch", type="int", default=30, help="Number of Epoch")
    parser.add_option("-a", "--augmen", dest="augmen", action="store_true", default=True, help="Augemntation?")
    parser.add_option("-l", "--lrate", dest="lr", default=0.002, type="float", help="set learing rate.")
    parser.add_option("-b", "--batchSize", dest="batchSize", type="int", default=4, help="Batch Size.")

    options, args = parser.parse_args()

    #%% Option store

    dataHome = options.dataHome
    preTrain = options.preTrain # ["sqxcep", "sqz"]
    aug = options.augmen
    resFold = options.result
    lr = options.lr
    nb_epoch = options.nb_epoch
    batchSize = options.batchSize
    sensor = options.sensor
    
    
#    heatmap = options.heatmap
    trFolder = os.path.expanduser(os.path.join('~', dataHome, sensor))
    
    # Make Validation folder
    trFolderList = trFolder.split('/')
    trFolderList[-2] = 'Val'
    trFolderList[-1] = trFolderList[-1].replace('Train', 'Val')
    trFolderList.insert(0,'/')
    valFolder = reduce(os.path.join, trFolderList)
    
    target_size = lvd.imageSizeLoader(trFolder)
    modelSaveFolder = os.path.expanduser(os.path.join('~', dataHome, resFold))
#
    labels = lvd.binaryLabel(bg=False)
    label = lvd.labelOut(labels)

    classNum = len(label)

    samples_per_epoch = lvd.getSamplePerEpoch(trFolder, label)

    nb_val_samples = lvd.getSamplePerEpoch(valFolder, label)
#%%
    if preTrain == 'gsl':
        model = gram.shallowGramModel(input_shape=target_size)
    elif preTrain == 'gml':
        model = gram.get_MultipleGramModel(input_shape=target_size)
    elif preTrain == 'gvs':
        model = gram.fireGramModel(input_shape=target_size)
    elif preTrain == 'gss':
        model = gram.get_smallGramModel(input_shape=target_size)
    else:
        print("Can not find your model. check your preTrain option")
        os._exit(1) 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###    #%%    
    prefix = ''
    if aug:
        prefix += "Aug"
        train_datagen = ImageDataGenerator(
                           rescale = 1./255, 
                           horizontal_flip=True,
                           vertical_flip=True,
                           rotation_range=30,
                           width_shift_range=0.20,
                           height_shift_range=0.20,
                           shear_range=0.2,
                           zoom_range=0.2,
                           fill_mode= 'reflect'
                           )
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
##    
    train_generator = train_datagen.flow_from_directory(
        trFolder,
        classes = label,
        target_size=target_size,
        batch_size=batchSize,
        color_mode = "grayscale")
    val_generator = val_datagen.flow_from_directory(
        valFolder,
        classes = label,
        target_size=target_size,
        batch_size=batchSize,
        color_mode = "grayscale")
    
#%%    
##    #%%
##    # Save file name setting    
    nameSplit = sensor.split('/')
    year = nameSplit[1][-2:]
    sen = nameSplit[-1][:3]
##
    saveFileName = preTrain+"_"+prefix+year+'_'+sen
    saveModelName = saveFileName+'_'+"{epoch:02d}-{val_acc:.2f}"
    saveFolder = os.path.expanduser(os.path.join('~',dataHome, resFold))
    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    modelSaveName = os.path.join(saveFolder,saveModelName+'.hdf5')
    modelLogName = os.path.join(saveFolder, saveFileName+'.log')
    modelConfJson = os.path.join(saveFolder,saveFileName+'.json')
#
    reduceLr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
    checkpoint = ModelCheckpoint(modelSaveName, monitor='val_loss', 
             verbose=1, save_best_only=True, mode='min')
    # if you run model continously you have to specify modelLogName your own logName
    csv_logger = CSVLogger(modelLogName, append=True)
    callbacks_list = [checkpoint, csv_logger, reduceLr]
#
    model.compile(optimizer=Adamax(lr = lr), loss='categorical_crossentropy', metrics=['accuracy'])
#

    log = model.fit_generator(train_generator,
        steps_per_epoch = samples_per_epoch // batchSize,
        epochs = nb_epoch,
        validation_data=val_generator,
        validation_steps = nb_val_samples // batchSize,
        callbacks=callbacks_list
        )
#

    json_model = model.to_json()
    lvd.jsonDumper(modelConfJson, json_model)
    lvd.makeResultFigure(modelLogName)

    modelHistroy = os.path.join(saveFolder,saveFileName+"_histroy"+'.json')
    lvd.jsonDumper(modelHistroy, str(log.history))






