import os
from hparams import EPOCH, LR, BATCH_SIZE
from data_generation import train_data, train_labels, validation_data,validation_labels
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.callbacks
from keras.models import Model
from keras import layers, losses

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("DATA IMPORTED")


# define model structure
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(8000, 1)),
            layers.Conv1D(16, 25, padding='same', strides=1, activation='relu'),
            layers.AveragePooling1D(pool_size=2, strides=2, padding='same'),
            layers.Conv1D(32, 25, padding='same', strides=1, activation='relu'),
            layers.AveragePooling1D(pool_size=2, strides=2, padding='same'),
            layers.Conv1D(64, 25, padding='same', strides=1, activation='relu'),
            layers.AveragePooling1D(pool_size=2, strides=2, padding='same'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(64, 25, padding='same', strides=1, activation='relu'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(32, 25, padding='same', strides=1, activation='relu'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(16, 25, padding='same', strides=1, activation='relu'),
            layers.UpSampling1D(size=2),
            layers.Conv1DTranspose(1, 25, padding='same', strides=1, activation='sigmoid'),
        ])
        print("MODEL SETUP & DATA IMPORTED")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# compile the model
autoencoder = Denoise()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), loss=losses.MeanSquaredError())
print("LOSS FUNCTION & OPTIMIZER DONE")

# TRAIN THE MODEL
checkpoint_cb = keras.callbacks.ModelCheckpoint("./model/autoencoder",
                                                save_best_only=True,
                                                # verbose=0,
                                                save_format='tf')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = autoencoder.fit(train_data, train_labels,
                          epochs=EPOCH,
                          batch_size=BATCH_SIZE,
                          # validation_split=0.25,
                          validation_data=(validation_data, validation_labels),
                          shuffle=True,
                          verbose=1,
                           callbacks=[checkpoint_cb, early_stopping_cb
                                     # , tensorboard_cb
                                      ]
                          )
print(autoencoder.encoder.summary(), autoencoder.decoder.summary())
print("MODEL TRAINING FINISHED")

# Save trained parameters
autoencoder.encoder.save("./model/encoder.h5")
autoencoder.decoder.save("./model/decoder.h5")
file = open('./model/history.pkl', 'wb')
pickle.dump(history.history, file)
file.close()
print("MODEL SAVED")

# Plot loss
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel('Epoch',fontdict={'weight': 'normal', 'size': 15})
plt.ylabel('Loss',fontdict={'weight': 'normal', 'size': 15})
plt.title('Model Loss',fontsize=15)
plt.legend(['Training Loss','Validation Loss'],fontsize=13)
plt.show()
