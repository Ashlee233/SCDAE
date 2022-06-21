import keras
import matplotlib
from data_generation import test_data1,test_labels,test_data,max_number_x,min_number_x,max_number_y,min_number_y,max_number_z,min_number_z
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# load model
autoencoder = keras.models.load_model("./model/autoencoder")
decoded_data = autoencoder.predict(test_data)
print("test data have been input")

# calculate mse
n = np.size(test_data, 0)
mse = np.zeros((n), dtype=float)
for i in range(n):
    mse[i] = tf.losses.mean_squared_error(decoded_data[i].squeeze(), test_labels[i].squeeze())
plt.bar(range(n), mse)
plt.title('MSE',fontsize=15)
plt.show()

### normalization
test_labels_x=test_data1[81]
test_labels_y=test_data1[82]
test_labels_z=test_data1[83]
test_labelsdata=np.concatenate((test_labels[81],test_labels[82],test_labels[83]), axis=0)

test_data_x=test_data[81]*(max_number_x-min_number_x)+min_number_x
test_data_y=test_data[82]*(max_number_y-min_number_y)+min_number_y
test_data_z=test_data[83]*(max_number_z-min_number_z)+min_number_z

decoded_data_x=decoded_data[81]*(max_number_x-min_number_x)+min_number_x
decoded_data_y=decoded_data[82]*(max_number_y-min_number_y)+min_number_y
decoded_data_z=decoded_data[83]*(max_number_z-min_number_z)+min_number_z
denoisy_cdae=np.concatenate((decoded_data[81],decoded_data[82],decoded_data[83]), axis=0)

SNR_cdae = 10*np.log10((np.sum(test_labelsdata**2))/(np.sum((denoisy_cdae-test_labelsdata)**2)))
RMSE_CDAE = (metrics.mean_squared_error(test_labelsdata, denoisy_cdae))**0.5
print('\nSNRin=20dB: \nSNR_cdae=',SNR_cdae,'\nRMSE-cdae=',RMSE_CDAE)


### plot
# test_labels
x_sample = np.reshape(test_labels_x,(1,8000))
x_sample = x_sample[0]
y_sample = np.reshape(test_labels_y,(1,8000))
y_sample = y_sample[0]
z_sample = np.reshape(test_labels_z,(1,8000))
z_sample = z_sample[0]

# noisy data
x_noise = np.reshape(test_data_x,(1,8000))
x_noise = x_noise[0]
y_noise = np.reshape(test_data_y,(1,8000))
y_noise = y_noise[0]
z_noise = np.reshape(test_data_z,(1,8000))
z_noise = z_noise[0]

# denoised data
x_denoise = np.reshape(decoded_data_x,(1,8000))
x_denoise = x_denoise[0]
y_denoise = np.reshape(decoded_data_y,(1,8000))
y_denoise = y_denoise[0]
z_denoise = np.reshape(decoded_data_z,(1,8000))
z_denoise = z_denoise[0]

matplotlib.rcParams['axes.unicode_minus']=False

### The phase-space portraits of the Lorenz attractor
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8,5))
ax=plt.subplot(111)
ax.scatter(x_sample[0:len(x_sample)-10],x_sample[10:len(x_sample)],s=0)
plt.plot(x_sample[0:len(x_sample)-10],x_sample[10:len(x_sample)])
ax.set_xlabel('x(t)',fontsize=17)
ax.set_ylabel('x(t+0.1)',fontsize=17)
plt.title('Clean signal',fontsize=15)
plt.figure(figsize=(8,5))
ax=plt.subplot(111)
ax.scatter(x_noise[0:len(x_sample)-10],x_noise[10:len(x_noise)],s=0)
plt.plot(x_noise[0:len(x_sample)-10],x_noise[10:len(x_noise)])
ax.set_xlabel('x(t)',fontsize=17)
ax.set_ylabel('x(t+0.1)',fontsize=17)
plt.title('Noisy signal with SNR_in = 20dB',fontsize=15)
plt.figure(figsize=(8,5))
ax=plt.subplot(111)
ax.scatter(x_denoise[0:len(x_sample)-10],x_denoise[10:len(x_denoise)],s=0)
plt.plot(x_denoise[0:len(x_sample)-10],x_denoise[10:len(x_denoise)])
ax.set_xlabel('x(t)',fontsize=17)
ax.set_ylabel('x(t+0.1)',fontsize=17)
plt.title('Signal denoised by proposed SCDAE',fontsize=15)
# plt.show()

### 1D comparison chart
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# clean
plt.figure(figsize=(15, 5))
plt.xlim(0, 8000)
plt.plot(np.arange(len(test_labels[30])), test_labels[30])
plt.xlabel('Time',fontdict={'weight': 'normal', 'size': 20})
plt.ylabel('Amplitude',fontdict={'weight': 'normal', 'size': 20})
plt.title('Clean signal',fontsize=18)
# noisy
plt.figure(figsize=(15, 5))
plt.xlim(0, 8000)
plt.plot(np.arange(len(test_data[30])), test_data[30])
plt.xlabel('Time',fontdict={'weight': 'normal', 'size': 20})
plt.ylabel('Amplitude',fontdict={'weight': 'normal', 'size': 20})
plt.title('Noisy signal with SNR_in = 10dB',fontsize=18)
# denoising
plt.figure(figsize=(15, 5))
plt.xlim(0, 8000)
plt.plot(np.arange(len(decoded_data[30])), decoded_data[30])
plt.xlabel('Time',fontdict={'weight': 'normal', 'size': 20})
plt.ylabel('Amplitude',fontdict={'weight': 'normal', 'size': 20})
plt.title('Signal denoised by proposed SCDAE',fontsize=18)
plt.show()