#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:16:48 2016

@author: park
"""
#%%
#from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, AveragePooling2D
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator
import datetime
import os
from keras.optimizers import Nadam
#from numpy import array_str
import lvdUtil.liveUtil as lvd
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
#import tensorflow as tf
import json
#%%
# My Own layer
def vgg_top_Model(x, classNum):
    x = GlobalAveragePooling2D(name='gavg_pool')(x) # It can changed to GolbalMaxPooling
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classNum, activation='softmax', name='predictions')(x)
    return x

def Xception_top_Model(x, classNum):
    x = GlobalAveragePooling2D(name='gavg_pool')(x)
    x = Dense(classNum, activation='softmax', name='predictions')(x)
    return x
    
def ResNet50_top_Model(x, classNum):
    x = GlobalAveragePooling2D(name='agvg_pool')(x)
#    x = Flatten()(x)
    x = Dense(classNum, activation='softmax', name='predictions')(x)
    return x
    
def InceptionV3_top_Model(x, classNum):
#    x = GlobalAveragePooling2D(name='agvg_pool')(x)
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(classNum, activation='softmax', name='predictions')(x)
    return x

def jsonDumper(fileName, jsonString):
    with open(fileName, 'w') as f:
        f.write(json.dumps(jsonString))
    return 

    
# Main function 
#%%
if __name__=="__main__":
#%% Option Parser
###################################################
## uncomment me ###################################   
###################################################
    use = "Usage : %prog [option]"
    parser = OptionParser(usage=use)

    parser.add_option("-d","--dataHome", dest='dataHome',
                  default="data/SegM", help="Home directory of data")
    parser.add_option("-s", "--sensor", dest="sensor",
                  default="LivDet2013/CrossMatchTrain", help="Sensor folder")
    parser.add_option("-p", "--pretrain", dest="preTrain", default="res", 
                      help="Pretraining Model")
    parser.add_option("-b", "--binary", dest="binary",
                      default=True, help="Binary Class?")
    parser.add_option("-r", "--resultFolder", dest="result",
                      default="models/origin", help="saving folder")
    parser.add_option("-e", "--epoch", dest="nb_epoch", default=1, help="Number of Epoch")
    parser.add_option("-a", "--augmen", dest="augmen", default = False, help="Augemntation?")
    parser.add_option("-f", "--fine", dest="fine", default = False, help="Augemntation?")
    # parser.add_option("-g", "--gpu", dest="gpu", default = 0, help= "GPU select")
    
    options, args = parser.parse_args()

    #%% Option store

    sensor = options.sensor
    dataHome = options.dataHome
    preTrain = options.preTrain
    binary = options.binary
    aug = options.augmen
    fine = options.fine
    resFold = options.result
    # gpu = "/gpu:%s" % options.gpu 
    
    if preTrain not in ['vgg', 'exc', 'res', 'inc']:
        print("Pretrain option must be one of ['vgg', 'exc', 'res', 'inc']")
        raise ValueError
    if options.binary: category = 'twoClass'
    else: category = 'manyClass'
    nb_epoch = int(options.nb_epoch)
       
    tempFolder = os.path.join(dataHome, 'Train', sensor, category)
    trainFolder = os.path.expanduser(os.path.join('~', tempFolder))
    tempFolder = os.path.join(dataHome, 'Val', sensor, category)
    valFolder = os.path.expanduser(os.path.join('~', tempFolder))
    
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
    if preTrain == 'vgg':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(weights='imagenet', include_top=False)
        predictions = vgg_top_Model(base_model.output, classNum)
        
    elif preTrain == 'exc':
        from keras.applications.xception import Xception
        base_model = Xception(weights='imagenet', include_top=False)
        predictions = Xception_top_Model(base_model.output, classNum)
        
    elif preTrain == 'res':
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False)
        predictions = ResNet50_top_Model(base_model.output, classNum)
        
    elif preTrain == 'inc':
        from preModels.inceptionV3 import InceptionV3
        model = InceptionV3(weights='imagenet', include_top=True, classNum=classNum)
#        predictions = InceptionV3_top_Model(base_model.output, classNum)
    else:
        print("There is no adequate Pretrain option.")
        raise ValueError
#%% 
    if preTrain != 'inc':       
        model = Model(input=base_model.input, output=predictions)
        #%%
        # first: train only the top layers (which were randomly initialized)
    ############################################################################
        for layer in base_model.layers:
            layer.trainable = False
        target_size = (224, 224)
    else: # This part is only for Inception V3 (Size problem)
        for layer in model.layers[:-1]:
            layer.trainable = False
        target_size = (299, 299)
    ############################################################    
        # compile the model (should be done *after* setting layers to non-trainable)

#%%
    # this is the augmentation configuration we will use for training
    prefix = ''
    if aug:
        prefix += "Aug"
        train_datagen = ImageDataGenerator(
                           rescale = 1./255, 
                           horizontal_flip=True,
                           rotation_range=30,
                           width_shift_range=0.20,
                           height_shift_range=0.20,
                           shear_range=0.3,
                           zoom_range=0.3,
                           fill_mode= 'reflect'
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
    # Training Start
    print("%s models is starting." % preTrain)

   
    if not fine:
            #%% Save file name setting    
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
        checkpoint = ModelCheckpoint(modelSaveName, monitor='val_loss', 
                 verbose=1, save_best_only=True, mode='min')
        csv_logger = CSVLogger(modelLogName, append=True)
        callbacks_list = [checkpoint, csv_logger, reduceLr]

        model.compile(optimizer=Nadam(lr=0.0005), loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        
        log = model.fit_generator(train_generator,
            samples_per_epoch = samples_per_epoch,
            nb_epoch = nb_epoch,
            validation_data=val_generator,
            nb_val_samples = nb_val_samples,
            callbacks=callbacks_list)
        json_model = model.to_json()
        jsonDumper(modelConfJson, json_model)
    else:
        if nb_epoch < 10:
            print ("You need more epochs")
            raise
        prefix+="Fine"
        nb_epoch = nb_epoch - 5
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
        
        reduceLr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5)
        checkpoint = ModelCheckpoint(modelSaveName, monitor='val_acc', 
                 verbose=1, save_best_only=True, mode='max')
        csv_logger = CSVLogger(modelLogName, append=True)
        callbacks_list = [checkpoint, csv_logger, reduceLr]
        
        model.compile(optimizer=Nadam(lr=0.002), loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        
        log = model.fit_generator(train_generator,
            samples_per_epoch = samples_per_epoch,
            nb_epoch = 5,
            validation_data=val_generator,
            nb_val_samples = nb_val_samples)
        
        for layer in model.layers:
            layer.trainable = True
            
        model.compile(optimizer=Nadam(lr=0.0002), loss='categorical_crossentropy', 
                  metrics=['accuracy'])
        
        log = model.fit_generator(train_generator,
            samples_per_epoch = samples_per_epoch,
            nb_epoch = nb_epoch,
            validation_data=val_generator,
            nb_val_samples = nb_val_samples,
            callbacks=callbacks_list)
        json_model = model.to_json()
        jsonDumper(modelConfJson, json_model)
    
    del model


    



