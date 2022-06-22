import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, InputLayer, LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import L1, L2, L1L2
# from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


def average_fft(fft, num_bins):
    num_samples_per_bin = int(len(fft) / num_bins)
    fft = fft[: num_samples_per_bin * num_bins]
    averaged_fft = np.average(fft.reshape(-1, num_samples_per_bin), axis=1)
    return averaged_fft

def truncate_fft(fft, truncating_point):
    return fft[: truncating_point]

def reshape_fft(fft):
    fft = np.expand_dims(fft, axis=0)
    fft = np.expand_dims(fft, axis=2)
    return fft

def preprocess_fft(fft, num_bins, truncating_point):
    truncated_fft = np.abs(truncate_fft(fft, truncating_point))
    averaged_fft = average_fft(truncated_fft, num_bins)

    return averaged_fft

def preprocess_stft(stft, num_bins, truncating_point, num_ffts_in_stft):
    preprocessed_stft = np.empty([num_bins, num_ffts_in_stft])
    for i in range(num_ffts_in_stft):
        preprocessed_stft[:, i] = preprocess_fft(stft[:, i, :], num_bins, truncating_point)
    
    preprocessed_stft = np.expand_dims(preprocessed_stft, 2)
    return preprocessed_stft

def remove_nan_values_from_dataset(inputs, outputs):
    input_contains_nan_counter = 0
    severe_nan_counter = 0
    between_nan_counter = 0
    mild_nan_counter = 0
    indices_to_delete = []

    for input_index in range(inputs.shape[0]):

        # check if this input example contains a nan value
        if np.isnan(inputs[input_index]).any():
            input_contains_nan_counter += 1

            if outputs[input_index] == 'severe':
                severe_nan_counter += 1
            elif outputs[input_index] == 'mild':
                mild_nan_counter += 1
            else:
                between_nan_counter += 1

            indices_to_delete.append(input_index)


    print(f'Total number of input examples containing a nan value: {input_contains_nan_counter}')
    print(f'Number of input examples containing a nan value from the "severe" class: {severe_nan_counter}')
    print(f'Number of input examples containing a nan value from the "between" class: {between_nan_counter}')
    print(f'Number of input examples containing a nan value from the "mild" class: {mild_nan_counter}\n')

    inputs = np.delete(inputs, indices_to_delete, 0)
    outputs = np.delete(outputs, indices_to_delete, 0)
    print(f'Inputs still contain nan values: {np.isnan(inputs).any()}\n')
    print(f'Shape of input data (without nan values): {inputs.shape}')
    print(f'Shape of output data (without nan values): {outputs.shape}\n')

    return inputs, outputs

def print_class_balance(inputs, outputs):
    severe_counter = 0
    mild_counter = 0
    between_counter = 0

    for input_index in range(inputs.shape[0]):

        if outputs[input_index] == 'severe':
            severe_counter += 1
        elif outputs[input_index] == 'mild':
            mild_counter += 1
        else:
            between_counter += 1

    print(f'Number of input examples from the "severe" class: {severe_counter}')
    print(f'Number of input examples from the "between" class: {between_counter}')
    print(f'Number of input examples from the "mild" class: {mild_counter}')


def get_train_and_val_data(inputs, outputs, num_classes, train_index, validation_index):

    # onehot_encoder = OneHotEncoder(sparse=False)
    # outputs = np.array(outputs).reshape(len(outputs), 1)
    # onehot_outputs = onehot_encoder.fit_transform(outputs)
    # X_train, X_val = inputs[train_index], inputs[validation_index]
    # y_train, y_val = onehot_outputs[train_index], onehot_outputs[validation_index]

    label_encoder = LabelEncoder()
    integer_outputs = label_encoder.fit_transform(outputs)
    X_train, X_val = inputs[train_index], inputs[validation_index]
    y_train, y_val = integer_outputs[train_index], integer_outputs[validation_index]
    
    return X_train, y_train, X_val, y_val


def create_and_compile_model(input_shape, output_dim, learning_rate):
    model = build_small_conv_model(input_shape, output_dim)

    opt = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, callbacks, evaluate_on_test_set=False, x_test=None, y_test=None):
    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks,
                        verbose=0)
    
    if evaluate_on_test_set:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return history, loss, accuracy


def plot_model_histories(model_histories, save_image_path):
    plt.rcParams['figure.figsize'] = [25, 13]
    plt.figure()

    for history_index in range(len(model_histories)):
        plt.subplot(2, 2, 1)
        plt.plot(model_histories[history_index].history['accuracy'], label=f'Fold no. {history_index+1}')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        
        plt.subplot(2, 2, 2)
        plt.plot(model_histories[history_index].history['val_accuracy'], label=f'Fold no. {history_index+1}')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(model_histories[history_index].history['loss'], label=f'Fold no. {history_index+1}')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(model_histories[history_index].history['val_loss'], label=f'Fold no. {history_index+1}')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

    plt.show()
  

def build_small_conv_model(input_shape, number_of_classes):

    kernel_size = (4, 4)
    max_pooling_dims = (2, 2)

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(4, kernel_size, activation=LeakyReLU(), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(max_pooling_dims, strides=max_pooling_dims, padding='same'))

    model.add(Conv2D(8, kernel_size, activation=LeakyReLU(), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(max_pooling_dims, strides=max_pooling_dims, padding='same'))

    model.add(Conv2D(16, kernel_size, activation=LeakyReLU(), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(max_pooling_dims, strides=max_pooling_dims, padding='same'))

    model.add(Conv2D(32, kernel_size, activation=LeakyReLU(), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(max_pooling_dims, strides=max_pooling_dims, padding='same'))

    model.add(Conv2D(32, kernel_size, activation=LeakyReLU(), padding='same'))
    # model.add(Dropout(0.1))
    model.add(MaxPooling2D(max_pooling_dims, strides=max_pooling_dims, padding='same'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(32))
    model.add(Dense(16))
    # model.add(Dense(8))

    model.add(Dense(number_of_classes, activation='softmax'))

    return model


def build_experimental_conv_model(input_shape, number_of_classes):

    num_filters_1 = 8
    num_filters_2 = 16
    num_filters_3_4 = 32
    kernel_size = 2
    l1_rate = 2e-6
    l2_rate = 2e-6
    dropout_rate = 0.2
    num_nodes_dense = 16

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(num_filters_1, (kernel_size, kernel_size), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(num_filters_2, (kernel_size, kernel_size), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(num_filters_3_4, (kernel_size, kernel_size), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (kernel_size, kernel_size), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(num_filters_3_4, (kernel_size, kernel_size), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(8))

    model.add(Dense(number_of_classes, activation='softmax', name='visualized_layer'))

    return model



def get_binary_class_occurrences(y_train, y_val):
    one_of_the_two = y_train[0]
    the_other_one = y_train[5]

    y_train_1_0 = 0
    for sample_index in range(len(y_train)):
        if (y_train[sample_index] == one_of_the_two).all():
            y_train_1_0 = y_train_1_0 + 1

    print(f'{y_train_1_0} occurrences of {one_of_the_two} in the training set')
    print(f'{y_train.shape[0] - y_train_1_0} occurrences of {the_other_one} in the training set')
    
    
    y_val_1_0 = 0
    for sample_index in range(len(y_val)):
        if (y_val[sample_index] == one_of_the_two).all():
            y_val_1_0 = y_val_1_0 + 1

    print(f'{y_val_1_0} occurrences of {one_of_the_two} in the validation set')
    print(f'{y_val.shape[0] - y_val_1_0} occurrences of {the_other_one} in the validation set')    

