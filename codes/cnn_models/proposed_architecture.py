#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:19:58 2018

@author: selen  $conda activate selen_keras
"""

from __future__ import print_function, division
import numpy as np
np.random.seed(42)
import tensorflow
tensorflow.compat.v1.random.set_random_seed(42)
import os
import glob
import itertools
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, confusion_matrix

import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from tqdm import tqdm  
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import time

'''
@brief: returns 3D RGB image to 2D grayscale
 '''  
# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# def load_data(classes, path, img_width, img_height):
#     data = []
#     data_id = []
#     data_labels = []

#     for fld in classes:        
#         index = classes.index(fld)  
#         combined_path = os.path.join(path, fld, '*png')
#         desc = 'Loading {} files (Index: {})'.format(fld, index, combined_path)
#         files = glob.glob(combined_path)  
#         for fl in tqdm(files, desc=desc):       
#             flbase = os.path.basename(fl)
            
#             #   load images
#             img = cv2.imread(fl)
#             img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
            
#             data.append(img)         
#             data_id.append(flbase)  
#             data_labels.append(index)             


#     return data, data_labels, data_id

def load_image_mat(classes, path, img_width, img_height):
    data = []
    data_id = []
    data_labels = []
    
    for fld in classes:
        index = classes.index(fld)
        combined_path = os.path.join(path, fld, '*png')
        desc = 'Loading {} files (Index: {})'.format(fld, index)
        files = glob.glob(combined_path)  
        for fl in tqdm(files, desc=desc):   
            flbase = os.path.basename(fl)
            
            d = plt.imread(fl)
#            d = d[:,:,0:3] # use if img eis RGB
           
            data.append(d)
            data_id.append(flbase)
            data_labels.append(index)
    
    print("1\n")
    data = np.array(data, dtype=np.float32)
    print("data.mean(): ",data.mean())
    print("data.std(): ",data.std())
                                                                      
    data = (data - data.mean()) / data.std()    # Normalization
    print("3\n")
    data = data[..., None]

    print("4\n")
    data_labels = np.array(data_labels, dtype=np.uint8)
    print("5\n")
    data_labels = np_utils.to_categorical(data_labels, len(classes))
    print('\nShape of data:', data.shape)
    
    return data, data_labels, data_id

# def normalize_data(data, data_label, num_classes):
#     #train_data, train_target, train_id = load_train(classes, train_path, img_width, img_height) 
#     data = np.array(data, dtype=np.uint8) 
#     data_label = np.array(data_label, dtype=np.uint8) 
#     data = data.transpose((0, 1, 2, 3)) 
#     data = data.astype('float32')
#     data = data / 255
#     data_label = np_utils.to_categorical(data_label, num_classes)

#     print('\nShape of data:', data.shape)
#     return data, data_label

      
#%%
def build_classifier(img_width, img_height, num_classes):
    classifier = Sequential()   


    # This gives 91% accuracy for 3 class with extended data, 222k trainable params
#    classifier.add(Conv2D(16, (3, 3), input_shape = (img_height, img_width, 3), kernel_initializer='random_uniform'))
#    classifier.add(Activation('relu'))    
#    classifier.add(Conv2D(16, (3, 3), input_shape = (img_height, img_width, 3), kernel_initializer='random_uniform'))
#    classifier.add(Activation('relu'))
#    classifier.add(BatchNormalization())
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))

#    classifier.add(Conv2D(16, (3, 3), input_shape = (img_height, img_width, 3), kernel_initializer='random_uniform'))
#    classifier.add(Activation('relu'))
#    classifier.add(Conv2D(16, (3, 3), input_shape = (img_height, img_width, 3), kernel_initializer='random_uniform'))
#    classifier.add(Activation('relu'))
#    classifier.add(BatchNormalization())
#    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(16, (3, 3), input_shape=(128, 128, 1), kernel_initializer='random_uniform'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))    

    classifier.add(Conv2D(16, (3, 3), kernel_initializer='random_uniform')) 
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))    
    
    classifier.add(Conv2D(16, (3, 3), kernel_initializer='random_uniform'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Conv2D(16, (3, 3), kernel_initializer='random_uniform'))#, kernel_regularizer=regularizers.l2(0.001)))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    classifier.add(Flatten())
    
    classifier.add(Dense(units = 16, kernel_initializer='random_uniform'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.5))#

#    classifier.add(Dense(units = 64, kernel_initializer='random_uniform'))
#    classifier.add(BatchNormalization())
#    classifier.add(Activation('relu'))
#    classifier.add(Dropout(0.5))#
    
    classifier.add(Dense(units = num_classes, kernel_initializer='random_uniform'))
    classifier.add(BatchNormalization())
    classifier.add(Activation('softmax'))
    ####################################################################
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    #####################################################################
    classifier.compile(optimizer = adam, loss= 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

#%%
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import warnings
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            
            
callbacks = [
    EarlyStoppingByAccuracy(monitor='val_acc', value=0.906, verbose=1),
]

def run_train(n_folds, train_data, train_target, train_batch_size, nb_epoch, img_width, img_height, num_classes):
    num_fold = 0
    models = []
    model_accs= []
    histories = []
    kf = KFold(n_folds, shuffle=True, random_state=42)
    
    for train_index, test_index in kf.split(train_data):
        classifier = build_classifier(img_width=img_width,
                              img_height=img_height,
                              num_classes=num_classes
                              )
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]
        num_fold += 1
        print('Training on fold {} of {}...'.format(num_fold, n_folds))
        print('Size of train split: ', len(X_train), len(Y_train))
        print('Size of validation split: ', len(X_valid), len(Y_valid))
        
        history = classifier.fit(X_train,
                      Y_train,
                      batch_size=train_batch_size,    # for SGD mini-batch size
                      epochs=nb_epoch,
                      shuffle=True,
                      verbose=1,
                      validation_data=(X_valid, Y_valid))
                      #callbacks=callbacks)
        
        predictions_valid = classifier.predict(X_valid.astype('float32'), batch_size=train_batch_size, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Loss for fold {0}: '.format(num_fold), score)
        #sum_score += score*len(test_index)
        models.append(classifier)
        histories.append(history)

        evalResult = classifier.evaluate(X_valid, Y_valid)
        model_accs.append(evalResult[1])    # get accuracy
        break # for debug purpose!

    cnt = 0
    for i in model_accs:
        print("Evaluation Accuracy for fold {}: {}%".format(cnt, i))
        cnt = cnt + 1
    
        
    print("Max Accuracy: {}% on fold {}".format(max(model_accs), model_accs.index(max(model_accs))))
    #score = sum_score/len(train_data)
    #print("Average loss across folds: ", score)
    info_string = "{0}fold_{1}x{2}_{3}epoch".format(n_folds, img_width, img_height, nb_epoch)
    
    return info_string, models, model_accs.index(max(model_accs)), histories

#%%
def plot_test_confusion_matrix(cls_pred, labels_test, classes):
    cls_true = [classes[np.argmax(x)] for x in labels_test]
    print("cls_true: ", len(cls_true))
    print("cls_pred: ", len(cls_pred))
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred,
                          labels=classes)

    print("cm: ", cm)

    plt.matshow(cm)  #'RdBu' , 'cubehelix', 'Blues'
    plt.colorbar() 
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35)  
    plt.yticks(tick_marks, classes)
   
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    
    plt.xlabel('Predicted')
    plt.ylabel('True')


    plt.savefig('confusionmatrix_less_data_3class', format='svg')   
    
    plt.show()
   


def print_test_accuracy(test_model, test_batch_size, sample_test, labels_test, classes, show_example_errors=False, show_confusion_matrix=False):
    
    test_batch_size =test_batch_size
    num_test = len(labels_test)    
    cls_pred = np.zeros(shape=num_test, dtype=np.int)    
    i = 0
    start = time.time()
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = sample_test[i:j, :]    
        cls_pred[i:j] = [np.argmax(x) for x in test_model.predict(images)]  
        i = j
    end = time.time()
    
    print("PREDICT TIME: ", end - start)

    cls_pred = np.array([classes[x] for x in cls_pred])  

    cls_true = np.array([classes[np.argmax(x)] for x in labels_test])

    correct = (cls_true == cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on test set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))


    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_test_confusion_matrix(cls_pred=cls_pred,
                                   labels_test=labels_test,
                                   classes=classes
                                   )
        
    return float(acc)


#%%
def plot_history(histories, key='categorical_crossentropy', plot_type='loss'):
  plt.figure(figsize=(16,10))
  val_type = ''
  train_type = ''
  if plot_type=='loss':
    val_type = 'val_loss'
    train_type = 'loss'
  else:
    val_type = 'val_acc'
    train_type = 'acc'

  for name, history in histories:
    val = plt.plot(history.epoch, history.history[val_type],
                   '--', label=name.title()+val_type)
    plt.plot(history.epoch, history.history[train_type], color=val[0].get_color(),
             label=name.title()+train_type)

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.savefig('extended_rgb_noiir_{}.svg'.format(train_type), format='svg') 
        
#%%

#Hyper Parameters
# trainPath = '/media/rcetin/42375AE8738FEA25/newdata_060819/noiir/data/train' 
# testPath = '/media/rcetin/42375AE8738FEA25/newdata_060819/noiir/data/test'

trainPath = '/home/selen/Desktop/grayscale_all/grayscale_all_resnet/train' 
testPath = '/home/selen/Desktop/grayscale_all/grayscale_all_resnet/test'

classNames = ["b", "g", "n"]#, "b_less", "g_less", "n_less"] 

batchSize = 32
imgWidth, imgHeight = 128, 128 
numChannels = 1
nFolds = 5
nbEpoch = 100
        
#%%

# Load train and test data
print("Preparing Train and Test Data. \nTrain Path: {}\nTest Path: {}".format(trainPath, testPath))

trainData, trainTarget, trainId = load_image_mat(classes=classNames,
                                            path=trainPath,
                                            img_width=imgWidth,
                                            img_height=imgHeight
                                            )

#trainData, trainTarget, trainId = load_data(classes=classNames,
#                                            path=trainPath,
#                                            img_width=imgWidth,
#                                            img_height=imgHeight
#                                            )
#
#trainData, trainTarget = normalize_data(data=trainData,
#                                        data_label=trainTarget,
#                                        num_classes=len(classNames)
#                                        )
#
print("Preparing Test Data...")
testData, testTarget, testId = load_image_mat(classes=classNames,
                                            path=testPath,
                                            img_width=imgWidth,
                                            img_height=imgHeight
                                            )

#testData, testTarget, testId = load_data(classes=classNames,
#                                            path=testPath,
#                                            img_width=imgWidth,
#                                            img_height=imgHeight
#                                            )
#
#testData, testTarget = normalize_data(data=testData,
#                                        data_label=testTarget,
#                                        num_classes=len(classNames)
#                                        )

#%%

model_final = build_classifier(img_width=imgWidth,
                      img_height=imgHeight,
                      num_classes=len(classNames)
                      )

model_final.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])

mcp_save = ModelCheckpoint('/tmp/EnetB0_CIFAR10_TL.h5', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

#print("Training....")
model_final.fit(trainData, trainTarget,
              batch_size=batchSize,
              epochs=nbEpoch,
              validation_split=0.1,
              callbacks=[mcp_save, reduce_lr],
              shuffle=True,
              verbose=1)


# train model
# info_string, models, best_index, histories = run_train(n_folds=nFolds, 
#                                             train_data=trainData, 
#                                             train_target=trainTarget, 
#                                             train_batch_size=batchSize, 
#                                             nb_epoch=nbEpoch, 
#                                             img_width=imgWidth, 
#                                             img_height=imgHeight,
#                                             num_classes=len(classNames)
#                                             )

# model = models[best_index]
# model = random.choice(models)

#plot histories
# plot_history([('', histories[best_index])], plot_type='loss')
# plot_history([('', histories[best_index])], key='Accuracy (%)', plot_type='acc')

# shuffle test data
# testData, testTarget = shuffle(testData, testTarget, random_state=0)  

# #make test

# accuracy = print_test_accuracy(test_model=model, 
#                     test_batch_size=batchSize, 
#                     sample_test=testData, 
#                     labels_test=testTarget,
#                     classes=classNames,
#                     show_example_errors=False, 
#                     show_confusion_matrix=True, 
#                     )  

testData, testTarget = shuffle(testData, testTarget, random_state=0)  

accuracy = print_test_accuracy(test_model=model_final, 
                    test_batch_size=batchSize, 
                    sample_test=testData, 
                    labels_test=testTarget,
                    classes=classNames,
                    show_example_errors=False, 
                    show_confusion_matrix=True, 
                    )  



#%%
model_final.summary()

#%%

# serialize model to JSON
model_json = model_final.to_json()
with open("grayscale_proposed_second_training_{0:.1%}.json".format(accuracy), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
w_file = "grayscale_proposed_second_training_{0:.1%}.h5".format(accuracy)
model_final.save_weights(w_file, overwrite=True)
print("Saved model to disk as grayscale_proposed_second_training_*.h5")

#%%

path = "/home/selen/Desktop/esl_paper/proposed/grayscale_proposed_second_training_87.1%"

#load json and create model
json_file = open("{}.json".format(path), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("{}.h5".format(path))
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#loaded_model.compile(optimizer = sgd, loss= 'categorical_crossentropy', metrics = ['accuracy'])
print_test_accuracy(test_model=loaded_model, 
                    test_batch_size=batchSize, 
                    sample_test=testData, 
                    labels_test=testTarget,
                    classes=classNames,
                    show_example_errors=False, 
                    show_confusion_matrix=True, 
                    )    


#%%%

from keras.models import load_model
from keras_flops import get_flops

## Calculate FLOPS
loaded_model.save(path)

model = load_model(path)

flops = get_flops(model, batch_size=batchSize)
print(f"FLOPS: {flops / 10 ** 9:.03} G")





