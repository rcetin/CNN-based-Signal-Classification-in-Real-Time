import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
from sklearn.utils import shuffle
import cv2

from keras.models import load_model, model_from_json
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tqdm import tqdm  
import os
import glob
import itertools
import cv2
from sklearn.metrics import log_loss, confusion_matrix


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
#            d = rgb2gray(d) # use if img eis grayscale
            #d = (d - d.mean()) / d.std()
           # d *= 255
            
            data.append(d)
            data_id.append(flbase)
            data_labels.append(index)
    
    data = np.array(data, dtype=np.float32)
#    data = stats.zscore(data)
    data = (data - data.mean()) / data.std()    # Normalization
    print("data.shape: ", np.shape(data))
    data = data[..., None]
    print("data.shape2: ", np.shape(data))

    data_labels = np.array(data_labels, dtype=np.uint8)
    data_labels = np_utils.to_categorical(data_labels, len(classes))
    
    return data, data_labels, data_id



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
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = sample_test[i:j, :]    
        cls_pred[i:j] = [np.argmax(x) for x in test_model.predict(images)]  
        i = j
    

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

img_height = 128
img_width = 128

base_model = applications.MobileNetV2(weights= None, include_top=False, input_shape= (img_height,img_width,1))

x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)

predictions = Dense(3, activation="softmax")(x)

model_final = Model(inputs = base_model.input, outputs = predictions)

model_final.summary()

#%%

#Hyper Parameters
# trainPath = '/media/rcetin/42375AE8738FEA25/newdata_060819/noiir/data/train' 
# testPath = '/media/rcetin/42375AE8738FEA25/newdata_060819/noiir/data/test'

# trainPath = '/home/selen/Desktop/rgb_all/data_rgb_v1/train' 
# testPath = '/home/selen/Desktop/rgb_all/data_rgb_v1/test'

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

print("Preparing Test Data...")
testData, testTarget, testId = load_image_mat(classes=classNames,
                                            path=testPath,
                                            img_width=imgWidth,
                                            img_height=imgHeight
                                            )

#%%

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

#%%

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

# serialize model to JSON
model_json = model_final.to_json()
with open("grayscale_mobilenetV2_{0:.1%}.json".format(accuracy), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
w_file = "grayscale_mobilenetV2_{0:.1%}.h5".format(accuracy)
model_final.save_weights(w_file, overwrite=True)
print("Saved model to disk as grayscale_mobilenetV2_*.h5")

#%%

path = "/home/selen/Desktop/esl_paper/mobilenetV2/grayscale_mobilenetV2_89.2%"

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


