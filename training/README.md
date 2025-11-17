# Training

> Python + TensorFlow (Keras)

Our goal is to train a basic Artificial Neural Network (`ANN`) of the `MLP` (MultiLayer Perceptron) type,
to run it afterward on really low-end hardware.

MLP neural networks are trained using the backpropagation technique, and they're mainly used to classify images
into multiple categories.
We train one of them using our laptop and a dataset of images with a correct label assigned.
The various online tutorials use [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) dataset, which is a
collection of `60,000` `28x28 px` images of handwritten digits (e.g. `2` or `3`) with an associated integer label.

There is only one issue: **memory**.
After the training phase, we need to keep the weights in the final executable, and the Tensorflow example written
in their documentation requires a little bit more than 1MB of disk space.
Input layer is `784` pixels (`28x28`), hidden layer 1 has a size of `700`, hidden layer 2 has a size of `500`
and output size is `10` (`0, 1, 2, ..., 9`).
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 700)                 │          35,000 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation (Activation)              │ (None, 700)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 700)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 500)                 │         350,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_1 (Activation)            │ (None, 500)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 500)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           5,010 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_2 (Activation)            │ (None, 10)                  │               0 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 390,510 (1.49 MB)
 Trainable params: 390,510 (1.49 MB)
 Non-trainable params: 0 (0.00 B)
```

Considering `4` bytes (`32 bits`) for each floating point weight, this means that the final weights will have a total
size of `1.49MB`.
Due to the hardware limitations of our NES (`40KB` cartridge: `32KB` for code and `8KB` for graphic assets, [source](https://en.wikipedia.org/wiki/Memory_management_controller_(Nintendo))),
we can't afford to put over 1MB of constants inside it.
Later, the 40KB limit has been extended to 1MB thanks to [Mapper chips](https://www.nesdev.org/wiki/Mapper) inside the
cartridge, but in this experiment I'll try to avoid them.

I resized the images in the training dataset to `7x7` (`49` pixels) and added two hidden layers of neurons with sizes of
`49` and `24`, keeping the output size of `10`. This reduced accuracy but also the memory required.
```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 49)                  │           2,450 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation (Activation)              │ (None, 49)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 49)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 24)                  │           1,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_1 (Activation)            │ (None, 24)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 24)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │             250 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_2 (Activation)            │ (None, 10)                  │               0 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,900 (15.23 KB)
 Trainable params: 3,900 (15.23 KB)
 Non-trainable params: 0 (0.00 B)
```

Surprise! We have reduced the size of the neural network data by `100` times and now only `15.23KB`
(`3900` weights+biases `*` `4 bytes`) are required.
We can store them inside the ROM and still have enough space for the actual program instructions!
Does this make our AI a "toy"? Maybe, but our plan is to get it up and running, it doesn't need to be perfect.

## Run
### Requirements
- Docker, using Tensorflow CPU image
### Script
Run the training script inside a docker container:
```shell
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:2.20.0 python ./training.py
```

## Accuracy
```
Epoch 47/50
938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8172 - loss: 0.6183  
Epoch 48/50
938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8163 - loss: 0.6213  
Epoch 49/50
938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8164 - loss: 0.6217  
Epoch 50/50
938/938 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.8179 - loss: 0.6153  
157/157 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - accuracy: 0.8886 - loss: 0.3672   
Test accuracy: 88.9%
```

Using the test data provided with the dataset we achieved an accuracy of `88.9%`, which is really not bad for such a
small neural network. Basically, it guessed the correct digit in almost `9` out of `10` images.

## Resources
Inspired by:
- [TensorFlow Docs](https://www.tensorflow.org/guide/core/mlp_core)
- [EloquentArduino Blog](https://eloquentarduino.github.io/2020/11/tinyml-on-arduino-and-stm32-cnn-convolutional-neural-network-example/).
- [Oddly Satisfying Deep Learning](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/mlp_keras.html)
- [V7Labs Blog](https://www.v7labs.com/blog/neural-networks-activation-functions)
