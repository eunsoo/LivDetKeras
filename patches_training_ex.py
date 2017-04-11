# -*- coding: utf-8 -*-


#from keras.preprocessing import image

"""
    This script is for squeezeNet having pretrained weight.
    In this code, only final convolution layer is trained first.
    After that, all layers are also trained along a few epoch.
"""
from keras.optimizers import Nadam
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from optparse import OptionParser
from keras.preprocessing.image import ImageDataGenerator
#import datetime
import os
import preModels.overSqueeze as ovs

import preModels.park_squeezenet as sq

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
                  default="Patch/Train/P32/LivDet2011/BiometrikaTrain/manyClass", help="Sensor folder")
#    parser.add_option("-s", "--sensor", dest="sensor",
#                  default="Patch/Train/P32/LivDet2013/BiometrikaTrain/manyClass", help="Sensor folder")
    parser.add_option("-p", "--pretrain", dest="preTrain", default="ssq", help="Pretraining Model")
    parser.add_option("-r", "--resultFolder", dest="result",
                      default="models/Patch/ssq", help="saving folder")
    parser.add_option("-e", "--epoch", dest="nb_epoch", type="int", default=1, help="Number of Epoch")
    parser.add_option("-a", "--augmen", dest="augmen", action="store_true", default=False, help="Augemntation?")
    parser.add_option("-m", "--base", dest="base", default = "models/ssq", help="base model")
    parser.add_option("-l", "--lrate", dest="lr", default=0.002, type="float", help="set learing rate.")
    parser.add_option("-b", "--bg", dest="background", action="store_false", default=True, help="Raw model(not pretrained.)")
    parser.add_option("-n", "--baseModel", dest="baseName", default="sa13_Bio_twoClass_11231838.hdf5", help="base model name.")
#    parser.add_option("-c", "--heatMap", dest="heatmap", action = "store_true", default=False, help="make heat map.")
    parser.add_option("-c", "--batchSize", dest="batchSize", type="int", default=192, help="Batch Size.")

    options, args = parser.parse_args()

    #%% Option store
    sensor = options.sensor
    dataHome = options.dataHome
    preTrain = options.preTrain # ["basic", "ssq"]
    aug = options.augmen
    resFold = options.result
    lr = options.lr
    backGround = options.background
    batchSize = options.batchSize
    nb_epoch = options.nb_epoch

    basemodel = os.path.expanduser(os.path.join('~', dataHome, options.base, options.baseName))
    
    trainFolder = os.path.expanduser(os.path.join('~',dataHome, sensor))
    valFolder = os.path.expanduser(os.path.join('~', dataHome, sensor.replace("Train","Val")))
 
    target_size = (int(sensor.split('/')[2][1:3]), int(sensor.split('/')[2][1:3]))
    modelSaveFolder = os.path.expanduser(os.path.join('~', dataHome))
   
    labels = lvd.binaryLabel(bg=backGround)
    label = lvd.labelOut(labels)

    classNum = len(label)

    samples_per_epoch = lvd.getSamplePerEpoch(trainFolder, label)

    nb_val_samples = lvd.getSamplePerEpoch(valFolder, label)

#%%
    model = ovs.convnet(network=preTrain, input_size=target_size[0])
  
    prefix = ''
    if aug:
        prefix += "_Aug_"
        train_datagen = ImageDataGenerator(
                           rescale = 1./255, 
                           horizontal_flip=True,
                           vertical_flip=True,
                           rotation_range=30,
                           width_shift_range=0.10,
                           height_shift_range=0.10,
                           shear_range=0.2,
                           zoom_range=0.2,
                           fill_mode= 'reflect'
                           )
    else:
        train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        trainFolder,
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
 
    nameSplit = sensor.split('/')
    year = nameSplit[3][-2:]
    sen = nameSplit[4][:3]
    patchName = nameSplit[2]

    saveFileName = preTrain+patchName+"_"+prefix+year+'_'+sen
    saveModelName = saveFileName+'_'+"{epoch:02d}-{val_acc:.2f}"
    saveFolder = os.path.expanduser(os.path.join('~',dataHome, resFold))
    if not os.path.isdir(saveFolder):
        os.makedirs(saveFolder)
    modelSaveName = os.path.join(saveFolder,saveModelName+'.hdf5')
    modelLogName = os.path.join(saveFolder, saveFileName+'.log')
    modelConfJson = os.path.join(saveFolder,saveFileName+'.json')

    reduceLr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, verbose=1)
    checkpoint = ModelCheckpoint(modelSaveName, monitor='val_loss', 
             verbose=1, save_best_only=True, mode='min')
    # if you run model continously you have to specify modelLogName your own logName
    csv_logger = CSVLogger(modelLogName, append=True)
    callbacks_list = [checkpoint, csv_logger, reduceLr]

    model.compile(optimizer=Nadam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    log = model.fit_generator(train_generator,
        steps_per_epoch = samples_per_epoch // batchSize,
        epochs = nb_epoch,
        validation_data=val_generator,
        validation_steps = nb_val_samples // batchSize,
        callbacks=callbacks_list
        )

    json_model = model.to_json()
    lvd.jsonDumper(modelConfJson, json_model)
    lvd.makeResultFigure(modelLogName)

    modelHistroy = os.path.join(saveFolder,saveFileName+"_histroy"+'.json')
    lvd.jsonDumper(modelHistroy, str(log.history))
    
    