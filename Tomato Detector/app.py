import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping

''' Dataset Processing'''

datagen = ImageDataGenerator(
    rescale= 1./255, 
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode= 'nearest', 
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'data/',
    target_size= (224,224),
    batch_size= 40,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    'data/',
    target_size=(224,224),
    batch_size= 32,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

Early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    start_from_epoch=5
)


'''Creating Neural Network'''

from keras._tf_keras.keras.models import Sequential,Model
from keras._tf_keras.keras.layers import Conv2D,Dropout,Dense,MaxPooling2D,Flatten,BatchNormalization,Activation,Add,Input

model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_data.num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    callbacks=[Early_stop]
)

model.save("trained1.h5")