import operator
from scipy import io
import numpy as np
from sklearn import preprocessing


def Gaussian_noise(signal, SNR):
    SNR+=38.5
    noise = np.random.randn(*signal.shape)  # *signal.shape
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    return signal+noise

### data generation
# train data
train_list=[]
for letter in 'xyz':
    for i in range(1,41):
        s=letter+format(i)
        sim_measurement = io.loadmat('./Lorenz_data/Lorenz_data.mat')[s]              #load data
        x_sample = sim_measurement[:, 0]
        y_sample = sim_measurement[:, 1]
        z_sample = sim_measurement[:, 2]

        train_list.append(np.asarray(x_sample))
        train_list.append(np.asarray(y_sample))
        train_list.append(np.asarray(z_sample))

# validation data
validation_list=[]
for letter in 'xyz':
    for i in range(1,11):
        s=letter+format(i)
        sim_measurement = io.loadmat('./Lorenz_data/Lorenz_validation_data.mat')[s]              #load data
        x_sample = sim_measurement[:, 0]
        y_sample = sim_measurement[:, 1]
        z_sample = sim_measurement[:, 2]

        validation_list.append(np.asarray(x_sample))
        validation_list.append(np.asarray(y_sample))
        validation_list.append(np.asarray(z_sample))

# test data
test_list=[]
for letter in 'xyz':
    for i in range(1,11):
        s=letter+format(i)
        sim_measurement = io.loadmat('./Lorenz_data/Lorenz_test_data.mat')[s]              #load data
        x_sample = sim_measurement[:, 0]
        y_sample = sim_measurement[:, 1]
        z_sample = sim_measurement[:, 2]

        test_list.append(np.asarray(x_sample))
        test_list.append(np.asarray(y_sample))
        test_list.append(np.asarray(z_sample))

Lorenz_data1=np.array(train_list)
validation_data1=np.array(validation_list)
test_data1=np.array(test_list)

min_index_x, min_number_x = min(enumerate(test_data1[81]), key=operator.itemgetter(1))
max_index_x, max_number_x = max(enumerate(test_data1[81]), key=operator.itemgetter(1))
min_index_y, min_number_y = min(enumerate(test_data1[82]), key=operator.itemgetter(1))
max_index_y, max_number_y = max(enumerate(test_data1[82]), key=operator.itemgetter(1))
min_index_z, min_number_z = min(enumerate(test_data1[83]), key=operator.itemgetter(1))
max_index_z, max_number_z = max(enumerate(test_data1[83]), key=operator.itemgetter(1))

### min-max normalization
# train data
x = np.size(Lorenz_data1, 0)
y = np.size(Lorenz_data1, 1)
chaos_data = np.zeros((x, y, 1), dtype=float)
chaos_temp_data = np.zeros(shape=(1, 1))
min_max_scaler = preprocessing.MinMaxScaler()
chaos_exp_data = np.expand_dims(Lorenz_data1, axis=2)
for i in range(x):
    chaos_data[i] = min_max_scaler.fit_transform(chaos_exp_data[i])
    chaos_temp_data = np.concatenate((chaos_temp_data, min_max_scaler.fit_transform(chaos_exp_data[i])), axis=0)
Lorenz_data2 = np.delete(chaos_temp_data, 0, axis=0)
train_labels=np.reshape(Lorenz_data2,(360,8000,1))

# validation data
x = np.size(validation_data1, 0)
y = np.size(validation_data1, 1)
chaos_data = np.zeros((x, y, 1), dtype=float)
chaos_temp_data = np.zeros(shape=(1, 1))
min_max_scaler = preprocessing.MinMaxScaler()
chaos_exp_data = np.expand_dims(validation_data1, axis=2)
for i in range(x):
    chaos_data[i] = min_max_scaler.fit_transform(chaos_exp_data[i])
    chaos_temp_data = np.concatenate((chaos_temp_data, min_max_scaler.fit_transform(chaos_exp_data[i])), axis=0)
validation_data2 = np.delete(chaos_temp_data, 0, axis=0)
validation_labels=np.reshape(validation_data2,(90,8000,1))

# test data
x = np.size(test_data1, 0)
y = np.size(test_data1, 1)
chaos_data = np.zeros((x, y, 1), dtype=float)
chaos_temp_data = np.zeros(shape=(1, 1))
min_max_scaler = preprocessing.MinMaxScaler()
chaos_exp_data = np.expand_dims(test_data1, axis=2)
for i in range(x):
    chaos_data[i] = min_max_scaler.fit_transform(chaos_exp_data[i])
    chaos_temp_data = np.concatenate((chaos_temp_data, min_max_scaler.fit_transform(chaos_exp_data[i])), axis=0)
test_data2 = np.delete(chaos_temp_data, 0, axis=0)
test_labels=np.reshape(test_data2,(90,8000,1))

### add noise
# train data
train_data=Gaussian_noise(train_labels[0:60],0)
train_data=np.concatenate((train_data,(Gaussian_noise(train_labels[60:120],5))), axis=0)
train_data=np.concatenate((train_data,(Gaussian_noise(train_labels[120:180],10))), axis=0)
train_data=np.concatenate((train_data,(Gaussian_noise(train_labels[180:240],15))), axis=0)
train_data=np.concatenate((train_data,(Gaussian_noise(train_labels[240:300],20))), axis=0)
train_data=np.concatenate((train_data,(Gaussian_noise(train_labels[300:360],25))), axis=0)

# validation data
validation_data=Gaussian_noise(validation_labels[0:15],0)
validation_data=np.concatenate((validation_data,(Gaussian_noise(validation_labels[15:30],5))), axis=0)
validation_data=np.concatenate((validation_data,(Gaussian_noise(validation_labels[30:45],10))), axis=0)
validation_data=np.concatenate((validation_data,(Gaussian_noise(validation_labels[45:60],15))), axis=0)
validation_data=np.concatenate((validation_data,(Gaussian_noise(validation_labels[60:75],20))), axis=0)
validation_data=np.concatenate((validation_data,(Gaussian_noise(validation_labels[75:90],25))), axis=0)

# test data
test_data=Gaussian_noise(test_labels[0:15],0)
test_data=np.concatenate((test_data,(Gaussian_noise(test_labels[15:30],5))), axis=0)
test_data=np.concatenate((test_data,(Gaussian_noise(test_labels[30:45],10))), axis=0)
test_data=np.concatenate((test_data,(Gaussian_noise(test_labels[45:60],15))), axis=0)
test_data=np.concatenate((test_data,(Gaussian_noise(test_labels[60:75],20))), axis=0)
test_data=np.concatenate((test_data,(Gaussian_noise(test_labels[75:90],25))), axis=0)