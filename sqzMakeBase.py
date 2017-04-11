# -*- coding: utf-8 -*-


#from keras.preprocessing import image

"""
    This script is for squeezeNet having pretrained weight.
    In this code, only final convolution layer is trained first.
    After that, all layers are also trained along a few epoch.
"""
from keras.optimizers import Nadam
from keras.layers import Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator
import datetime
import os

import preModels.park_squeezenet as sq

import lvdUtil.liveUtil as lvd
#import json

#%%
    
if __name__ == '__main__':
#%%
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)

    parser.add_option("-d","--dataHome", dest='dataHome',
                  default="data/SegM", help="Home directory of data")
    parser.add_option("-s", "--sensor", dest="sensor",
                  default="LivDet2013/BiometrikaTrain", help="Sensor folder")
    parser.add_option("-p", "--pretrain", dest="preTrain", default="sxr", 
                      help="Pretraining Model")
    parser.add_option("-b", "--binary", dest="binary",
                      default=True, help="Binary Class?")
    parser.add_option("-r", "--resultFolder", dest="result",
                      default="models/base", help="saving folder")
    parser.add_option("-e", "--epoch", dest="nb_epoch", default=15, help="Number of Epoch")
    parser.add_option("-a", "--augmen", dest="augmen", default = True, help="Augemntation?")
    # parser.add_option("-m", "--base", dest="base", default = "models/base", help="base model")
    
    options, args = parser.parse_args()

    #%% Option store

    sensor = options.sensor
    dataHome = options.dataHome
    preTrain = options.preTrain # ["sqxcep", "sqz"]
    binary = options.binary
    aug = options.augmen
    resFold = options.result
    
    # gpu = "/gpu:%s" % options.gpu 
    
    if options.binary: category = 'twoClass'
    else: category = 'manyClass'
    
    nb_epoch = int(options.nb_epoch)
    
    tempFolder = os.path.join(dataHome, 'Train', sensor, category)
    trainFolder = os.path.expanduser(os.path.join('~', tempFolder))
    tempFolder = os.path.join(dataHome, 'Val', sensor, category)
    valFolder = os.path.expanduser(os.path.join('~', tempFolder))
    
    target_size = (224, 224)
    modelSaveFolder = os.path.expanduser(os.path.join('~', dataHome))
    
    if binary:
        labels = lvd.binaryLabel()
    else:
        labels = lvd.loadLabelFromYear(sensor.split('/')[0], sensor.split('/')[1])
    label = lvd.labelOut(labels)
    
    classNum = len(label)
    
    samples_per_epoch = lvd.getSamplePerEpoch(trainFolder)
    nb_epoch = int(nb_epoch)
    nb_val_samples = lvd.getSamplePerEpoch(valFolder)
    #%%

    baseHome = os.path.expanduser(os.path.join('~', tempFolder))
    if preTrain == "sqz":
        model = sq.get_squeezenet(classNum, dim_ordering='tf')
    elif preTrain == "sqxcep":
        model = sq.get_squeezeXceNet(classNum, dim_ordering='tf')
    elif preTrain == "sxr":
        model = sq.get_sqXceNetRes(classNum, dim_ordering='tf')
    else:
        print("model name is wrong.")

    #%%    
    prefix = ''
    if aug:
        prefix += "Aug"
        train_datagen = ImageDataGenerator(
                           rescale = 1./255, 
                           horizontal_flip=True,
                           # rotation_range=30,
                           # width_shift_range=0.20,
                           # height_shift_range=0.20,
                           # shear_range=0.2,
                           # zoom_range=0.2,
                           # fill_mode= 'reflect'
                           )
        val_datagen = ImageDataGenerator(
                           rescale = 1./255,
                           horizontal_flip=True,
                           # rotation_range=30,
                           # width_shift_range=0.10,
                           # height_shift_range=0.10,
                           # shear_range=0.1,
                           # zoom_range=0.1,
                           # fill_mode= 'reflect'
                           )
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255)
        val_datagen = ImageDataGenerator(rescale = 1./255)
    
    train_generator = train_datagen.flow_from_directory(
        trainFolder,
        classes = label,
        target_size=target_size,
        batch_size=32)
    val_generator = val_datagen.flow_from_directory(
        valFolder,
        classes = label,
        target_size=target_size,
        batch_size=32)
    #%%
    # Save file name setting    
    nameSplit = sensor.split('/')
    year = nameSplit[0][-2:]
    sen = nameSplit[1][:3]
    now = datetime.datetime.now().strftime('%m%d%H%M')
    saveFileName = preTrain+prefix+year+'_'+sen+'_'+category+'_'+now
    modelSaveName = os.path.expanduser(os.path.join('~',dataHome,
                 resFold,saveFileName+'.hdf5'))
    modelLogName = os.path.expanduser(os.path.join('~',dataHome,
                 resFold,saveFileName+'.log'))
    modelConfJson = os.path.expanduser(os.path.join('~',dataHome,
                 resFold,saveFileName+'.json'))
    reduceLr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
    checkpoint = ModelCheckpoint(modelSaveName, monitor='val_acc', 
             verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger(modelLogName, append=True)
    callbacks_list = [checkpoint, csv_logger, reduceLr]

    # I can add residual mapping and batch norm. This does not require additional parameters
    model.compile(optimizer=Nadam(lr=0.0002), loss='categorical_crossentropy', 
              metrics=['accuracy'])
#        model.compile(optimizer="adam", loss='categorical_crossentropy', 
#                  metrics=['accuracy'])
    
    log = model.fit_generator(train_generator,
        samples_per_epoch = samples_per_epoch,
        nb_epoch = nb_epoch,
        validation_data=val_generator,
        nb_val_samples = nb_val_samples,
        callbacks=callbacks_list)

    json_model = model.to_json()
    lvd.jsonDumper(modelConfJson, json_model)


#    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Training Start





