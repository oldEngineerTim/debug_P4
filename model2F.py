
import cv2
import csv
#import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sklearn
# import tensorflow as tf
from sklearn.model_selection import train_test_split


correction = 0.1 #  0.25 parameter to tune
del_rate =  0.1 # 0.4 0.8
del_rate2  = 1.0
cut_value = 99   # 0.02  0.5

def RandomBrightness(image):
    # convert to HSV 
    RandomImage = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    # randomly generate brightness value
    # define range dark to bright
    random_bright = np.random.uniform(0.25,1.0)
    # Apply the brightness to V channel
    RandomImage[:,:,2] = RandomImage[:,:,2]*random_bright
    # back to RGB
    RandomImage = cv2.cvtColor(RandomImage,cv2.COLOR_HSV2RGB)
    return RandomImage

lines = []
lines1 = []
lines2 = []
lines3 = []
lines4 = []
lines5 = [] 

meaurements = []
# read counter clockwise data run
with open('./wdata/data12/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)        
with open('./wdata/data13/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines1.append(line)
        
with open('./wdata/data14/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines2.append(line)
        
with open('./wdata/data15/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines3.append(line)    

with open('./wdata/data16/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines4.append(line)   
        
with open('./wdata/data17/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines5.append(line)
        
        
images = []
measurements = []
count = 0
for line in lines:
    for i in range(3):
       
        source_path = line[i]   # read the middle, left, right images limit 2^15
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end="\r",)
        current_path='./wdata/data12/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :
            measurements.append(measurement+correction)
        #  Right CAM    
        else:
            measurements.append(measurement-correction)   
print("")      
for line in lines1:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end=" \r",)
        current_path='./wdata/data13/IMG/' + filename  
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM  
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  Right CAM   
        else:    
            measurements.append(measurement-correction)
    
print("")      
for line in lines2:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end=" \r",)
        current_path='./wdata/data14/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  right CAM    
        else:
            measurements.append(measurement-correction)    

            
print("")      
for line in lines3:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end=" \r",)
        current_path='./wdata/data15/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  right CAM    
        else:
            measurements.append(measurement-correction)
            
print("")      
for line in lines4:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end=" \r",)
        current_path='./wdata/data16/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  right CAM    
        else:
            measurements.append(measurement-correction)            
 
print("")
      
for line in lines5:
    for i in range(3):
        source_path = line[i]   # read the middle, left, right images
        source_path=source_path.replace("\\","/")
        filename = source_path.split('/')[-1]
        count +=1
        print(count,end=" \r",)
        current_path='./data/data17/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        # Center CAM
        if i == 0 :
            measurements.append(measurement)
        # Left CAM
        elif i == 1 :    
            measurements.append(measurement+correction)
        #  right CAM    
        else:
            measurements.append(measurement-correction)

            
augment_images, augmented_measurements = [],[]
print("")
for image,measurement in zip(images, measurements):
    if abs(measurement) > cut_value or np.random.random() > del_rate:
        image = RandomBrightness(image) # random brightness
        count +=1
        print(count,end="\r",)
        augment_images.append(image)
        augmented_measurements.append(measurement)
        augment_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)

print("") 
 # training   limit  2^16
X_train = np.array(augment_images)   
y_train = np.array(augmented_measurements)   

print("Loading Done")
  
        



import keras

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation,ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

def resize(img):
    import tensorflow
    return tensorflow.image.resize_images(img,(60,120))

# network
model = Sequential()

# Crop 70 pixels from the top of the image and 25 from the bottom
model.add(Cropping2D(cropping=((75, 25), (0, 0)),
                     input_shape=(160, 320, 3),
                     data_format="channels_last"))

# Resize the data
model.add(Lambda(resize))

# Normalize the data
model.add(Lambda(lambda x: (x/127.5) - 0.5))

model.add(Conv2D(3, (1, 1), padding='same'))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(16, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
model.add(ELU())

model.add(Flatten())
model.add(ELU())

model.add(Dense(512))
model.add(Dropout(.2))
model.add(ELU())

model.add(Dense(100))
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(10))
model.add(Dropout(.5))
model.add(ELU())

model.add(Dense(1))

adam = Adam(lr=1e-5)
model.compile(loss = 'mse',optimizer = 'adam',metrics =['accuracy']) 

earlystopper = EarlyStopping(patience =5, verbose =1)
checkpointer = ModelCheckpoint('model3_new_M11a_11a_E50_r1_1.h5', monitor ='val_loss',verbose=1,save_best_only=True)

hist_obj = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 20)
model.save('model3_new_RND_0p1_M11a_10a_E20_r3_1.h5')
    
model.summary()          
print(model.summary())

#visualize the model
modelobj = load_model('model3_new_RND_0p1_M11a_10a_E20_r3_1.h5')
plot_model(modelobj, to_file='model3_new_RND_0p1_M11a_10a_E20_r3_1.png')



