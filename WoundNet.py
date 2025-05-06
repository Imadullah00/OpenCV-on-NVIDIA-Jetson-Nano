import os
import numpy as np
import cv2
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize

import tensorflow as tf
import keras
from keras import backend as K

import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score
#from keras.optimizers import Adam
import matplotlib.pyplot as plt


np.random.seed(42)

def iou_metric(y_true, y_pred):
    smooth = 1e-6
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Apply threshold
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coefficient(y_true, y_pred, smooth=1):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Apply threshold
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def augment_image(image, mask):
    # Ensure consistent flipping
    flip = tf.random.uniform(shape=[]) > 0.5
    image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
    mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    # Adjust brightness & contrast only for the image
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    return image, mask

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(512,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(1024,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Now begins the expansive path of the 'U'
u6 = tf.keras.layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(512,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.BatchNormalization()(c6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.BatchNormalization()(c7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9,c1],axis=3)
c9 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64,(3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1), activation='sigmoid', kernel_initializer='he_normal', padding='same')(c9)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', dice_coefficient, iou_metric])

model.summary()


model.load_weights("C:\\Users\\ImadF\\Desktop\\AI_JETSON_NANO\\unet.weights.h5")


print("Model and weights loaded successfully.")

img = cv2.imread("C:\\Users\\ImadF\\Desktop\\AI_JETSON_NANO\\test_images\\fusc_0027.png")  # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
input_image = np.expand_dims(img, axis=0)  # (1, H, W, 3)

# Predict mask
pred_mask = model.predict(input_image)[0]

# Threshold
pred_mask = (pred_mask > 0.5).astype(np.uint8)

# Display
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Predicted Mask")
plt.imshow(pred_mask[:,:,0], cmap='gray')
plt.show()



cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read(0)
    if not ret:
        break

    # --- Segmentation processing ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (IMG_WIDTH, IMG_HEIGHT))
    input_image = np.expand_dims(resized, axis=0)
    pred_mask = model.predict(input_image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    segmented = cv2.resize(pred_mask[:,:,0], (frame.shape[1], frame.shape[0]))
    segmented_bgr = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(frame,frame,mask=segmented)

    # --- Display ---
    cv2.imshow('Original', frame)
    cv2.imshow('Segmented', segmented_bgr)
    cv2.moveWindow('Segmented', 1500,0)
    cv2.imshow('Final_Result', result)
    cv2.moveWindow('Final_Result', 1500,600)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()