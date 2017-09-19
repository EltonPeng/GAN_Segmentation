import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense
from keras.layers import LeakyReLU, Dropout, Activation, BatchNormalization
from keras.layers import Reshape, concatenate, Flatten
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from skimage.transform import resize
from skimage.io import imsave

from data import load_test_data, load_train_data

K.set_image_data_format('channels_last')

img_rows = 96
img_cols = 96

class GAN(object):
    def __init__(self, img_rows = 96, img_cols = 96):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.Segmentor = None
        self.Discriminator = None
        self.AdversialModel= None
        self.DiscriminitorModel = None
        self.Generator = None

    def get_segmentor_discriminator_adveriasal(self):
        dropout = 0.75
        channel_depth = 32

        s_inputs = Input((self.img_rows, self.img_cols, 1), name='raw_image_input')
        conv1 = Conv2D(32, (3,3), padding='same')(s_inputs)
        conv1 = BatchNormalization(momentum=0.9)(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = Dropout(dropout)(conv1)
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

        conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
        conv2 = BatchNormalization(momentum=0.9)(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = Dropout(dropout)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
        conv3 = BatchNormalization(momentum=0.9)(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = Dropout(dropout)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
        conv4 = BatchNormalization(momentum=0.9)(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = Dropout(dropout)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3,3), padding='same')(pool4)
        conv5 = LeakyReLU(alpha=0.2)(conv5)

        up6 = Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv5)
        up6 = concatenate([up6, conv4], axis=3)
        conv6 = Conv2D(256, (3,3), padding='same')(up6)
        conv6 = BatchNormalization(momentum=0.9)(conv6)
        conv6 = LeakyReLU(alpha=0.2)(conv6)
        conv6 = Dropout(dropout)(conv6)

        up7 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(conv6)
        up7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(128, (3,3), padding='same')(up7)
        conv7 = BatchNormalization(momentum=0.9)(conv7)
        conv7 = LeakyReLU(alpha=0.2)(conv7)
        conv7 = Dropout(dropout)(conv7)

        up8 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(conv7)
        up8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(64, (3,3), padding='same')(up8)
        conv8 = BatchNormalization(momentum=0.9)(conv8)
        conv8 = LeakyReLU(alpha=0.2)(conv8)
        conv8 = Dropout(dropout)(conv8)

        up9 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(conv8)
        up9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(32, (3,3), padding='same')(up9)
        conv9 = BatchNormalization(momentum=0.9)(conv9)
        conv9 = LeakyReLU(alpha=0.2)(conv9)
        conv9 = Dropout(dropout)(conv9)

        conv10 = Conv2D(1, (1,1), name='segmentor_output')(conv9)

        d_conv1 = Conv2D(channel_depth * 1, 5, strides=2, padding='same')
        d_conv1_relu = LeakyReLU(alpha=0.2)
        d_conv1_dropout = Dropout(dropout)

        d_conv2 = Conv2D(channel_depth * 2, 5, strides=2, padding='same')
        d_conv2_relu = LeakyReLU(alpha=0.2)
        d_conv2_droput = Dropout(dropout)

        d_conv3 = Conv2D(channel_depth * 4, 5, strides=2, padding='same')
        d_conv3_relu = LeakyReLU(alpha=0.2)
        d_conv3_dropout = Dropout(dropout)

        d_conv4 = Conv2D(channel_depth * 8, 5, strides=1, padding='same')
        d_conv4_relu = LeakyReLU(alpha=0.2)
        d_conv4_dropout = Dropout(dropout)

        d_flatten = Flatten()
        d_logit = Dense(1)
        d_pred = Activation('sigmoid', name='discriminator_output')

        discriminator_input = Input(shape=(self.img_rows, self.img_cols, 2))
        discriminator_conv1 = d_conv1(discriminator_input)
        discriminator_conv1 = d_conv1_relu(discriminator_conv1)
        discriminator_conv1 = d_conv1_dropout(discriminator_conv1)

        discriminator_conv2 = d_conv2(discriminator_conv1)
        discriminator_conv2 = d_conv2_relu(discriminator_conv2)
        discriminator_conv2 = d_conv2_droput(discriminator_conv2)

        discriminator_conv3 = d_conv3(discriminator_conv2)
        discriminator_conv3 = d_conv3_relu(discriminator_conv3)
        discriminator_conv3 = d_conv3_dropout(discriminator_conv3)

        discriminator_conv4 = d_conv4(discriminator_conv3)
        discriminator_conv4 = d_conv4_relu(discriminator_conv4)
        discriminator_conv4 = d_conv4_dropout(discriminator_conv4)

        discriminator_flatten = d_flatten(discriminator_conv4)
        discriminator_logit = d_logit(discriminator_flatten)
        discriminator_pred = d_pred(discriminator_logit)



        adveriasal_d_input = concatenate([s_inputs, conv10], axis=3)
        adveriasal_d_conv1 = d_conv1(adveriasal_d_input)
        adveriasal_d_conv1 = d_conv1_relu(adveriasal_d_conv1)
        adveriasal_d_conv1 = d_conv1_dropout(adveriasal_d_conv1)

        adveriasal_d_conv2 = d_conv2(adveriasal_d_conv1)
        adveriasal_d_conv2 = d_conv2_relu(adveriasal_d_conv2)
        adveriasal_d_conv2 = d_conv2_droput(adveriasal_d_conv2)

        adveriasal_d_conv3 = d_conv3(adveriasal_d_conv2)
        adveriasal_d_conv3 = d_conv3_relu(adveriasal_d_conv3)
        adveriasal_d_conv3 = d_conv3_dropout(adveriasal_d_conv3)

        adveriasal_d_conv4 = d_conv4(adveriasal_d_conv3)
        adveriasal_d_conv4 = d_conv4_relu(adveriasal_d_conv4)
        adveriasal_d_conv4 = d_conv4_dropout(adveriasal_d_conv4)

        adveriasal_d_flatten = d_flatten(adveriasal_d_conv4)
        adveriasal_d_logit = d_logit(adveriasal_d_flatten)
        adveriasal_d_pred = d_pred(adveriasal_d_logit)

        segmentor = Model(inputs=[s_inputs], outputs=[conv10])
        discriminator = Model(inputs=[discriminator_input], outputs=[discriminator_pred])
        adveriasal_model = Model(inputs=[s_inputs], outputs=[adveriasal_d_pred])

        discriminator_optimizer = RMSprop(lr=0.0002, decay=6e-8)
        discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])

        adveriasal_optimizer = RMSprop(lr=0.0002, decay=6e-8)
        adveriasal_model.compile(loss='binary_crossentropy', optimizer=adveriasal_optimizer, metrics=['accuracy'])

        return segmentor, discriminator, adveriasal_model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def train_gan(batch_size = 256, epoch=1):

    print("*" * 30)
    print("loading and preprocessing train data")
    print("*" *30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = pre_apply_model_process_img(imgs_train)
    imgs_mask_train = pre_apply_model_process_img_mask(imgs_mask_train)

    gan = GAN()
    segmentor, discriminator, adveriasial_model = gan.get_segmentor_discriminator_adveriasal()

    for i in range(10 * (imgs_train.shape[0]//batch_size+1)):
        batch_index = np.random.randint(0, imgs_train.shape[0], batch_size)
        imgs_train_batch = imgs_train[batch_index, ...]
        imgs_mask_train_batch = imgs_mask_train[batch_index, ...]

        pred_mask = segmentor.predict(imgs_train_batch)

        discriminator_x_fake = np.concatenate([imgs_train_batch, pred_mask], axis=3)
        discriminator_x_true = np.concatenate([imgs_train_batch, imgs_mask_train_batch], axis=3)

        discriminator_x = np.concatenate([discriminator_x_true, discriminator_x_fake])

        discriminator_y = np.ones([2*batch_size, 1])
        discriminator_y[batch_size:,:] = 0

        discriminator_loss = discriminator.train_on_batch(discriminator_x, discriminator_y)

        discirminator_log_message = "%d: [Discriminator model loss: %f, accuracy: %f]" % (i, discriminator_loss[0], discriminator_loss[1])
        print(discirminator_log_message)

def pre_apply_model_process_img_mask(imgs_mask):
    imgs_mask = imgs_mask.astype('float32')
    imgs_mask /= 255.  # scale masks to [0, 1]
    return imgs_mask

def pre_apply_model_process_img(imgs):
    imgs = imgs.astype('float32')
    mean = np.mean(imgs)  # mean for data centering
    std = np.std(imgs)  # std for data normalization
    imgs -= mean
    imgs /= std
    return imgs

if __name__ == '__main__':
    train_gan()