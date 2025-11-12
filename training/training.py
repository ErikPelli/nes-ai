import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

if __name__ == "__main__":
    print("Tensorflow Version: " + tf.__version__)
    # Set random seed for reproducible results
    tf.random.set_seed(42) # Where's my towel?

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Compute the number of labels
    num_labels = len(np.unique(y_train))
    if num_labels != 10:
        raise Exception("Invalid number of dataset labels")

    # tf.image.resize requires a 4D Tensor with channel dimension, we add it here to the images
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Resize images from 28x28 to 7x7
    x_train = tf.image.resize(x_train, [7, 7]).numpy()
    x_test = tf.image.resize(x_test, [7, 7]).numpy()

    # Flatten (convert image data to 1D) the image
    # Normalize image pixels (from range [0,255] to range [0,1])
    input_size = 7*7
    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, input_size])
    x_test = x_test.astype('float32') / 255

    # Convert labeled values to one-hot vector
    # e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y_train = to_categorical(y_train, num_classes=num_labels)
    y_test = to_categorical(y_test, num_classes=num_labels)

    # Define model with Keras
    dropout_rate = 0.4
    hidden_layer_size = 24
    batch_size = 64

    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_layer_size))
    model.add(Activation("relu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_labels))
    model.add(Activation("softmax"))

    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=50, batch_size=batch_size)

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Test accuracy: %.1f%%" % (100.0 * acc))

    print("Exporting weights to file...")
    with open("model_weights.txt", "w") as f:
        for layer in model.layers:
            weights_list = layer.get_weights()

            # Skip empty lists (e.g. Activation and Dropout layers have no weights)
            if not weights_list:
                continue

            # In a Dense layer, get_weights() returns [kernel, bias]
            kernel, bias = weights_list[0], weights_list[1]
            num_inputs, num_neurons = kernel.shape

            # Export data to a file with this format: [bias, weight1, weight2, ...],
            # for each neuron, to make it compatible with the C code
            for neuron_index in range(num_neurons):
                f.write(f"{bias[neuron_index]}\n")

                for input_index in range(num_inputs):
                    weight = kernel[input_index, neuron_index]
                    f.write(f"{weight}\n")
    print("Weights saved in file model_weights.txt.")
