import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, InputLayer, LeakyReLU
from keras.optimizers import Adam
from keras.regularizers import L1, L2, L1L2
# from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


def get_train_and_val_data(inputs, outputs, train_index, validation_index):

    onehot_encoder = OneHotEncoder(sparse=False)
    outputs = np.array(outputs).reshape(len(outputs), 1)
    onehot_outputs = onehot_encoder.fit_transform(outputs)
    
    X_train, X_val = inputs[train_index], inputs[validation_index]
    y_train, y_val = onehot_outputs[train_index], onehot_outputs[validation_index]
    
    return X_train, y_train, X_val, y_val



def build_dense_model(input_shape, output_dim):
    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(2, activation = "relu"))
    model.add(Dense(4, activation = "relu"))
    model.add(Dense(8, activation = "relu"))
    model.add(Dense(16, activation = "relu"))
    # model.add(Dense(32, activation = "relu"))
    # model.add(Dense(64, activation = "relu"))
    # model.add(Dense(128, activation = "relu"))
    # model.add(Dense(256, activation = "relu"))
    # model.add(Dense(512, activation = "relu"))
    # model.add(Dense(1024, activation = "relu"))
    # model.add(Dense(512, activation = "relu"))
    # model.add(Dense(256, activation = "relu"))
    # model.add(Dense(128, activation = "relu"))
    # model.add(Dense(64, activation = "relu"))
    # model.add(Dense(32, activation = "relu"))
    # model.add(Dense(16, activation = "relu"))
    model.add(Dense(8, activation = "relu"))
    model.add(Dense(4, activation = "relu"))
    model.add(Dense(output_dim, activation = "softmax"))
    return model



def build_small_conv_model(input_shape, number_of_classes):

    l1_rate = 1e-5
    l2_rate = 1e-5
    dropout_rate = 0.5

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(8, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))

    model.add(Conv2D(16, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(16, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(8, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    

    # flatten output and feed it into dense layer
    model.add(Flatten())
    model.add(Dense(16))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(8))
    # model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(number_of_classes, activation='softmax'))

    return model



def build_large_conv_model(input_shape, number_of_classes):

    l1_rate = 1e-5
    l2_rate = 1e-5
    dropout_rate = 0.5

    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))

    model.add(Conv2D(16, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))

    model.add(Conv2D(32, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(128, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (3, 3), activation=LeakyReLU(), padding='same', kernel_regularizer=L1L2(l1=l1_rate, l2=l2_rate)))
    model.add(Dropout(dropout_rate))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same'))


    # flatten output and feed it into dense layer
    model.add(Flatten())
    model.add(Dense(32))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(16))
    # model.add(Dropout(dropout_rate))

    # output layer
    model.add(Dense(number_of_classes, activation='softmax'))

    return model



def create_and_compile_model(input_shape, model_type, output_dim, learning_rate):
    
    if model_type == 'dense':
        model = build_dense_model(input_shape, output_dim)
    else:
        model = build_small_conv_model(input_shape, output_dim)


    opt = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model



def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, evaluate_on_test_set=False, x_test=None, y_test=None):

    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0)
    
    if evaluate_on_test_set:
        model.evaluate(x_test, y_test, verbose=2)

    return history



def plot_model_histories(model_histories):
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



def plot_average_model_history(model_histories):

    average_training_accuracy = [0] * len(model_histories[0].history['accuracy'])
    average_validation_accuracy = [0] * len(model_histories[0].history['accuracy'])
    average_training_loss = [0] * len(model_histories[0].history['accuracy'])
    average_validation_loss = [0] * len(model_histories[0].history['accuracy'])

    for history_index in range(len(model_histories)):
        
        accuracy = model_histories[history_index].history['accuracy']
        val_accuracy = model_histories[history_index].history['val_accuracy']
        loss = model_histories[history_index].history['loss']
        val_loss = model_histories[history_index].history['val_loss']

        average_training_accuracy = [average_training_accuracy[i] + accuracy[i] for i in range(len(accuracy))]
        average_validation_accuracy = [average_validation_accuracy[i] + val_accuracy[i] for i in range(len(val_accuracy))]
        average_training_loss = [average_training_loss[i] + loss[i] for i in range(len(loss))]
        average_validation_loss = [average_validation_loss[i] + val_loss[i] for i in range(len(val_loss))]


    average_training_accuracy = [element / len(model_histories) for element in average_training_accuracy]
    average_validation_accuracy = [element / len(model_histories) for element in average_validation_accuracy]
    average_training_loss = [element / len(model_histories) for element in average_training_loss]
    average_validation_loss = [element / len(model_histories) for element in average_validation_loss]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(average_training_accuracy, label='Training')
    plt.plot(average_validation_accuracy, label='Validation')
    plt.title('Average Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(average_training_loss, label='Training')
    plt.plot(average_validation_loss, label='Validation')
    plt.title('Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()



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

