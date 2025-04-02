import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, MaxPooling2D, Add, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

########################### DATA PREPROCESSING ##########################
# Set paths here
DATASET_PATH = ''
TRAIN_PATH = DATASET_PATH + ''
TEST_PATH = DATASET_PATH + ''

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
NUM_CLASSES = 2

# Creating dataset generators
training_validation_data_gen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2],
    zoom_range=0.2,
    validation_split=0.2
)

testing_data_gen=ImageDataGenerator(
    rescale=1./255
)

# Loading dataset from directory
training_dataset=training_validation_data_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

validation_dataset=training_validation_data_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=True
)

testing_dataset=testing_data_gen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED,
    shuffle=True
)

# To check if categorical classes were recognized properly
print('Classes', list(training_dataset.class_indices))
print('Classes', list(validation_dataset.class_indices))
print('Classes', list(testing_dataset.class_indices))

##################### STARTING TO BUILD MODEL ################################
def shortcut_projection(x, filters, strides):
  projection = tf.keras.Sequential([
      Conv2D(filters, kernel_size=1, strides=strides, padding='same'),
      BatchNormalization()
  ])
  return projection(x)



def bottleneck_layers(x, kernel, strides, filters, projection_x):
  shortcut = x
  if projection_x is not None:
    shortcut = projection_x(x, filters*4, strides)

  # 1X1 REDUCING CONVOLUTION NETWORK
  x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  # KERNEL X KERNEL CONVOLUTION NETWORK
  x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)

  # 1X1 CONVOLUTION RESTORATION LAYER
  x = Conv2D(filters * 4, kernel_size=1, strides=1, padding='same')(x)
  x = BatchNormalization()(x)

  x = Add()([x, shortcut])
  x = ReLU()(x)

  return x



def resnet(input_shape, num_classes):
  inputs = Input(shape=input_shape)

  x = Conv2D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

  x = bottleneck_layers(x, kernel=3, strides=1, filters=64, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=128, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=256, projection_x=shortcut_projection)
  x = bottleneck_layers(x, kernel=3, strides=1, filters=256, projection_x=None)
  x = bottleneck_layers(x, kernel=3, strides=2, filters=512, projection_x=shortcut_projection)

  x = GlobalAveragePooling2D()(x)
  outputs = Dense(num_classes, activation='softmax')(x)

  model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
  return model



model = resnet(INPUT_SHAPE, NUM_CLASSES)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

########################## TRAINING THE MODEL #################################
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

EPOCHS = 50

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

###################### METRICS ##############################
_, test_accuracy = model.evaluate(testing_dataset)
print(f'Model accuracy: {test_accuracy}')

# Saving the model
model.save('CatDogResNet50.h5')