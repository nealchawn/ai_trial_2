# Building A neual Network Algorithm
# We ne to define the layers we want.
# First Layer is how many inputs
# Last Layer is the output or prediction layer

# How we want to use the Neural Network
# nn = NeuralNetwork.new(3,5,10) 3
# nn.train(training_data, training_labels)
#  expect output learning accuracy

# Finally either predict values based on array of training data
# predict(test_data)
# Or evaluate how well nn has done on given test_data and labels
# evaluate(test_data, test_labels)

# to build the network we'll need nodes
# to design the nodes, lets use them and then we'll know what
# we'll need to build later

# First thing our network needs is layers for nodes
# We define layers
# We define nodes

# Lets test our network to see it created the proper
# layers and nodes per layer

# Now We want to define the network prediction in our case the last layer
# node index will correspond to the prediction
# therefore the node with the highest output will be the prediction

# nodes will need to have outputs
# network will need to print out the index of the node in the
# last layer with the highest output

# we have yet to compute the outputs so for now
# I've initialized them to random

# When checking the network prediction we'll need to see all the output of
# all the nodes in the last layer to confirm it is correct

# in printing out the prediction theres 2-3 parts,
# one part may be any input we may eventually give
# the second part is functionally getting the max value from the last layer
# the third conditional part for checking is
# printing all outputs from the last layer, we do this to make sure
# the network prints out the correct max

# now it's time to train the network, so we'll setup a
# training function on the network, and then load-in the mnist data for testing and building.

# we need to set the node weights

# !pip install tensorflow keras numpy mnist matplotlib

# Local Imports
import sys
sys.path.append(".")
from neural_network import NeuralNetwork

import mnist
#import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.utils import to_categorical

# setup trainging and testing data
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# no need to normalize the data, it could be hard on the activation function we use
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

test = [2,3,4]
for index, tester in enumerate(test[1:]):
    print(tester)

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# initialize the network
#nn = NeuralNetwork([len(test_images[0]), 16,16, 10])
nn = NeuralNetwork([len(test_images[0]), 5, 10])
nn.train_network(train_images, train_labels)

# print the count of elements in the layers
for layer in nn.sizes:
    print(layer)

#data = list(zip(test_images[0:4], test_labels[0:4]))
# print outputs
#nn.predict(data)

print("Total Activations: ")
nn.total_activations()

print("try input and guess: test image 0")
nn.feedforward(test_images[0])
print(f"correct guess: {test_labels[0]}")
nn.network_outputs()
print("was it correct? ")

print("try input and guess: test image 1")
nn.feedforward(test_images[1])
print(f"correct guess: {test_labels[1]}")
nn.network_outputs()
print("was it correct? ")

print("try input and guess: test image 2")
nn.feedforward(test_images[2])
print(f"correct guess: {test_labels[2]}")
nn.network_outputs()
print("was it correct? ")


print("try input and guess: test image 2")
nn.feedforward(test_images[3])
print(f"correct guess: {test_labels[3]}")
nn.network_outputs()
print("was it correct? ")


import code;
code.interact(local=dict(globals(), **locals()))



# import code; code.interact(local=dict(globals(), **locals()))