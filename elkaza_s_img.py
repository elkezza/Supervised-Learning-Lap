# Solution for task 2 (Image Classifier) of lab assignment for FDA SS23 by SLAEH ELKAZA
# imports here
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# define additional functions here
def create_model(num_classes):
   # Creata  sequnatial modal objekt by Keras
    model = tf.keras.Sequential([
     # flatting X_train input (6336 features/column) into a 44x48x3 
        layers.Input(shape=(6336,)),
        #  4D tensor with dimensions (batch_size, height, width, channels).
        layers.Reshape((44, 48, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model




def train_predict(X_train, y_train, X_test):

    # check that the input has the correct shape
    assert X_train.shape == (len(X_train), 6336)
    assert y_train.shape == (len(y_train), 1)
    assert X_test.shape == (len(X_test), 6336)

    # --------------------------
    # add your data preprocessing, model definition, training and prediction between these lines

    # Normalize the images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode the labels
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train).toarray()

    # Spliet the training data into a training set and a validtaion set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=44)

    # Creat and train the model
    num_classes = y_train.shape[1]
    model = create_model(num_classes)
    #model = create_model()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=64)

    # Predikt the labals for the givan test data
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)


    # --------------------------

    # test that the returned prediction has correct shape
    assert y_pred.shape == (len(X_test),) or y_pred.shape == (len(X_test), 1)

    return y_pred


if __name__ == "__main__":
    # load data (please load data like that and let every processing step happen **inside** the train_predict function)
    # (change path if necessary)
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    # please put everything that you want to execute outside the function here!


# references:
# https://www.tensorflow.org/
# https://scikit-learn.org
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html