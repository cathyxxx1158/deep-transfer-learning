import tensorflow as tf
import os
import cv2
import keras,h5py
import shutil
import random
import pydicom
import numpy as np
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras.callbacks import Callback
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import torchxrayvision as xrv
from keras import layers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, InceptionResNetV2, DenseNet201, DenseNet121
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import Adam
from livelossplot.tf_keras import PlotLossesCallback

import keras.backend as K
from keras.models import Sequential


import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)

def walkFile(file,label,howmany,type,size=512):
    temp=[]
    tempy = []
    if type=="DICM":
        for root, dirs, files in os.walk(file):
            # root 表示当前正在访问的文件夹路径
            # dirs 表示该文件夹下的子目录名list
            # files 表示该文件夹下的文件list
            # 遍历文件
            i=0
            for i ,f in zip(tqdm(range(howmany)),files):
                    if i<howmany:
                        eachpath=os.path.join(root, f)
                        ds = pydicom.read_file(eachpath)
                        pix = cv2.resize(ds.pixel_array,(size,size))
                        temp.append(pix)
                        # plt.imshow(pix, cmap='gray')
                        # plt.show()
                        i+=1
    elif type=="JPEG":
        for root, dirs, files in os.walk(file):
            i = 0
            for i, f in zip(tqdm(range(howmany)), files):
                if i < howmany:
                    eachpath = os.path.join(root, f)
                    jpg=cv2.imread(eachpath,0)
                    pix = cv2.resize(jpg, (size, size))
                    temp.append(pix)
                    # plt.imshow(pix, cmap='gray')
                    # plt.show()
                    i += 1
    for j in range(0, howmany):
        tempy.append(label)
    return temp,tempy
    # ai32 = np.array(temp, dtype=np.uint8)
    # print("size of 0 int32 number: %f" % sys.getsizeof(ai32))
        #
        # # 遍历所有的文件夹
        # for d in dirs:
        #     print(os.path.join(root, d))

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return

def generate_partitionandlabel(file,howmany):
    temp=[]
    tempy = {}
    for labelindex,eachpath in zip(range(len(file)),file):
        for root, dirs, files in os.walk(eachpath):
            for i, f in zip(tqdm(range(howmany)), files):
                if i < howmany:
                    temp.append(f)
                    tempy[f]=labelindex
    random.shuffle(temp)
    return temp,tempy
# design a DataGenerator used for save memory

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(512, 512), n_channels=3,
                 n_classes=4, shuffle=True, filelist=[],split="train",mode="npy"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.filelist=filelist
        self.split=split
        self.mode=mode
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            IDs = os.path.splitext(ID)
            IDn = IDs[0]
            # Store class
            y[i] = self.labels[ID]
            # Store sample
            if self.labels[ID] == 0:
                X[i,]=np.load('F:/MLdata/Covid_npy/'+self.split+'/'+IDn+'.npy')
            else:
                if self.labels[ID] == 1:
                    X[i,] = np.load('F:/MLdata/Other_npy/' + self.split + '/PNEUMONIA/' + IDn + '.npy')
                elif self.labels[ID] == 2:
                    X[i,] = np.load('F:/MLdata/Other_npy/' + self.split + '/VIRUS/' + IDn + '.npy')
                else:
                    X[i,] = np.load('F:/MLdata/Other_npy/' + self.split + '/NORMAL/' + IDn + '.npy')

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

def pretrained_model1():

    pretrained_model1 = VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    for layer in pretrained_model1.layers[:12]:
        layer.trainable = False
    for layer in pretrained_model1.layers[13:]:
        layer.trainable = True

    model1 = Sequential()
    # first (and only) set of FC => RELU layers
    model1.add(AveragePooling2D((2, 2), name='avg_pool'))
    model1.add(Flatten())

    model1.add(Dense(64, activation='relu'))
    model1.add(Dropout(0.3))

    model1.add(Dense(4, activation='softmax'))

    preinput1 = pretrained_model1.input
    preoutput1 = pretrained_model1.output
    output1 = model1(preoutput1)
    model1 = Model(preinput1, output1)

    model1.summary()

    return model1

def pretrained_model2():
    pretrained_model2 = InceptionResNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    for layer in pretrained_model2.layers[:-280]:
        layer.trainable = False


    model2 = Sequential()

    model2.add(AveragePooling2D((2, 2), name='avg_pool'))
    model2.add(Flatten())
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.5))

    model2.add(Dense(64, activation='relu'))
    model2.add(Dropout(0.3))

    model2.add(Dense(4, activation='softmax'))

    preinput2 = pretrained_model2.input
    preoutput2 = pretrained_model2.output
    output2 = model2(preoutput2)
    model2 = Model(preinput2, output2)

    model2.summary()
    return model2

def pretrained_model3():
    pretrained_model3 = DenseNet121(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
    )
    for layer in pretrained_model3.layers[:-200]:
        layer.trainable = False

    model3 = Sequential()
    # first (and only) set of FC => RELU layers
    model3.add(AveragePooling2D((2, 2), name='avg_pool'))
    model3.add(Flatten())

    model3.add(Dense(64, activation='relu'))
    model3.add(Dropout(0.3))

    model3.add(Dense(4, activation='softmax'))

    preinput3= pretrained_model3.input
    preoutput3= pretrained_model3.output
    output3 = model3(preoutput3)
    model3 = Model(preinput3, output3)

    model3.summary()
    return model3

def make_model(name,input_shape,num_classes):
    if name == "VGG16":
        model = pretrained_model1()
    elif name == "InceptionResNetV2":
        model = pretrained_model2()
    elif name == "DenseNet121":
        model = pretrained_model3()

    return model

def trainmodel(model,savepath):
    # root_for_covid='F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/'
    # rootpath='F:/Edge_download/data/archive/chest_xray/chest_xray/'
    max_acc = 0
    if not os.path.exists('F:/MLdata/savedmodel/'+savepath):
        os.makedirs('F:/MLdata/savedmodel/'+savepath)
    root_for_covid = 'F:/MLdata/Covid/'
    rootpath = 'F:/MLdata/Other/'
    generate_dataset=(lambda split:[root_for_covid+split,rootpath+split+'PNEUMONIA/',rootpath+split+'VIRUS/',rootpath+split+'NORMAL/'])
    # trainset path
    trainset_filelist=generate_dataset('train/')
    # valset path
    valset_filelist=generate_dataset('val/')
    # testset path
    testset_filelist = generate_dataset('test/')
    partition_train,labels_train=generate_partitionandlabel(trainset_filelist,3000)
    training_generator = DataGenerator(partition_train, labels_train, batch_size=32, dim=(224, 224), n_channels=3,
                     n_classes=4, shuffle=True, filelist=trainset_filelist,split="train")
    #training_generator = ImageDataGenerator(rotation_range=50, width_shift_range = 0.2,height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True, fill_mode = 'nearest')
    partition_val,labels_val=generate_partitionandlabel(valset_filelist,3000)
    validation_generator = DataGenerator(partition_val, labels_val, batch_size=32, dim=(224, 224), n_channels=3,
                     n_classes=4, shuffle=True, filelist=valset_filelist,split="val")

    earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('F:/MLdata/savedmodel/'+savepath+'/save_at_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='auto')
    model.compile(optimizer=keras.optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"],)


    history = model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     steps_per_epoch=198,
                                     epochs=20,
                                     workers=8,
                                     max_queue_size=10, validation_steps=10,

                                     use_multiprocessing=False,
                                     callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
                                     )


    training_loss=history.history["loss"]
    train_acc=history.history["accuracy"]
    test_loss=history.history["val_loss"]
    test_acc=history.history["val_accuracy"]
    epoch_count=range(1,len(training_loss)+1)

    plt.plot(epoch_count,training_loss,'r--')
    plt.plot(epoch_count,test_loss,'b--')
    plt.legend(["Training_loss","Test_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.savefig('F:/MLdata/savedmodel/' + savepath + 'train_loss.png')
    plt.show()
    # X,y=validation_generator.__getitem__(1)

    plt.plot(epoch_count,train_acc,'r--')
    plt.plot(epoch_count,test_acc,'b--')
    plt.legend(["train_acc","test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.savefig('F:/MLdata/savedmodel/'+savepath+'test_acc.png')
    plt.show()

def testmodel(model, Weights):
    model.load_weights(Weights)
    root_for_covid = 'F:/MLdata/Covid/'
    rootpath = 'F:/MLdata/Other/'
    generate_dataset = (
        lambda split: [root_for_covid + split, rootpath + split + 'PNEUMONIA/', rootpath + split + 'VIRUS/',
                       rootpath + split + 'NORMAL/'])
    # testset path
    testset_filelist = generate_dataset('test/')
    X = np.empty((2110, 224,224, 3))
    y = np.empty((2110), dtype=int)
    partition_test,labels_test= generate_partitionandlabel(testset_filelist,3000)
    for ID,i in zip(partition_test,range(len(partition_test))):
        IDs = os.path.splitext(ID)
        IDn = IDs[0]
        y[i]=labels_test[ID]
        if labels_test[ID] == 0:
            X[i,] = np.load('F:/MLdata/Covid_npy/test/' + IDn + '.npy')
        else:
            if labels_test[ID] == 1:
                X[i,] = np.load('F:/MLdata/Other_npy/test/PNEUMONIA/' + IDn + '.npy')
            elif labels_test[ID] == 2:
                X[i,] = np.load('F:/MLdata/Other_npy/test/VIRUS/' + IDn + '.npy')
            else:
                X[i,] = np.load('F:/MLdata/Other_npy/test/NORMAL/' + IDn + '.npy')
    Y=keras.utils.to_categorical(y, num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    Score=model.evaluate(X,Y,verbose=0)
    print("Test loss",Score[0])
    print("Test accuracy",Score[1])
    # predictions = model.predict_classes(X)
    predictions = model.predict(X)
    predictions=np.argmax(predictions , axis=1)
    print(predictions[:15])
    predictions = predictions.reshape(1, -1)[0]
    print(predictions[:15])
    print(y[:15])
    print(classification_report(y, predictions, target_names=['Covid (Class 0)', 'PNEUMONIA (Class 1)','VIRUS (Class 2)','NORMAL (Class 3)']))



#trainmodel(model=make_model("InceptionResNetV2", input_shape=(224, 224, 3), num_classes=4), savepath="InceptionResNetV2")
#trainmodel(model= make_model("DenseNet121", input_shape=(224, 224, 3),num_classes=4), savepath="DenseNet121")
trainmodel(model= make_model("VGG16",input_shape=(224, 224,3),num_classes=4), savepath="VGG16")
# testmodel(model= make_model("InceptionResNetV2",input_shape=(224, 224,3),num_classes=4),Weights='F:/MLdata/savedmodel/InceptionResNetV2/save_at_15.h5')
# testmodel(model= make_model("VGG16",input_shape=(224, 224,3),num_classes=4),Weights='F:/MLdata/savedmodel/VGG16/save_at_15.h5')
# testmodel(model= make_model("DenseNet121",input_shape=(224, 224,3),num_classes=4),Weights='F:/MLdata/savedmodel/DenseNet121/save_at_15.h5')





