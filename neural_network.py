import numpy as np
import random

# Local Imports
import sys
sys.path.append(".")


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.activations = []
        self.outputs = []
        self.weights = []
        self.biases = []

        self.sizes = sizes
        self.set_random_weights()
        self.set_random_biases()

    def set_random_weights(self):
        for layer_index, layer_size in enumerate(self.sizes[1:], start=1):
            layer_weights = []
            for size in range(layer_size):
                for size in range(self.sizes[layer_index-1]):
                    layer_weights.append(random.uniform(-5.0, 5.0))
            self.weights.append(layer_weights)

    def set_random_biases(self):
        total_biases = 0
        # add extra zero bias to help future indexing
        #self.biases.append(0)
        for index, size in enumerate(self.sizes[0:-1], start=1):
            total_biases += 1
        for x in range(total_biases):
            self.biases.append(random.uniform(-5.0, 5.0))

    def train_network(self, training_data, training_labels):
        if len(training_data) != len(training_labels):
            print("Error data and labels must be the same length")
        data = list(zip(training_data, training_labels))
        self.sgd(data)

    def sgd(self, data, mini_batch_size = 1000):
        # first we'll create batches of training data

        n = len(data)
        data_batches = [
            data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        print(len(data_batches))
        i = 0
        for mini_batch in data_batches:
            print("Batch: " + str(i))
            i += 1
            self.update_mini_batch(mini_batch)
            self.network_outputs()

        print("Finished All training data!")

    def update_mini_batch(self, mini_data_batch):
        weight_gradients = []
        bias_gradients = []
        i = 0
        for training_input in mini_data_batch:
            training_object, training_label = training_input

            self.feedforward(training_object)

            weights_gradient, bias_gradient = self.backpropogation(training_label)

            weight_gradients.append(weights_gradient)
            bias_gradients.append(bias_gradient)

        # average gradients

        weights_gradient = np.average(weight_gradients,axis=0)
        biases_gradient = np.average(bias_gradients, axis=0)
        # may need to convert to list

        weights_gradient_list = []
        for weight_gradient in weights_gradient:
            weights_gradient_list.append(weight_gradient.tolist())

        #weights_gradient = weights_gradient.tolist()
        biases_gradient = biases_gradient.tolist()

        for x in range(len(self.biases)):
            self.biases[x] -= 0.1*biases_gradient[x]

        weight_gradient_index = 0
        for layer_index, layer_weights in enumerate(self.weights, start=0):
            for weight_index, weight in enumerate(layer_weights):
                self.weights[layer_index][weight_index] = weight - 0.1*weights_gradient_list[layer_index][weight_index]
                weight_gradient_index += 1


    def feedforward(self, training_object):
        # set inputs
        self.outputs = []
        self.activations = []

        temp_activations = []
        for index in range(self.sizes[0]):
            temp_activations.append(training_object[index])
        self.activations.append(temp_activations)

        for layer_index, layer_size in enumerate(self.sizes[1:], start=0):
            layer_weights = self.weights[layer_index]
            layer_inputs = self.activations[layer_index]
            weight_index = 0

            layer_outputs = []
            layer_activations = []

            for node_index in range(layer_size):
                node_weights = []
                # get node weights
                #print(f"layer size: {layer_size}, previous_layer_size: {self.sizes[layer_index]}, layer weights: {len(layer_weights)}")
                for x in range(self.sizes[layer_index]):
                    node_weights.append(layer_weights[weight_index])
                    weight_index += 1
                output = 0
                for indx in range(len(node_weights)):
                    output += layer_inputs[indx]*node_weights[indx]
                output = output + self.biases[layer_index]
                layer_outputs.append(output)
                layer_activations.append(self.sigmoid(output))
            self.outputs.append(layer_outputs)
            self.activations.append(layer_activations)

    def backpropogation(self, training_label):
        costs = []
        output_layer_activations = self.activations[-1]
        output_layer_outputs = self.outputs[-1]

        correct_labels = self.translate_label_to_array(training_label)

        costs.append(self.compute_cost_derivative(correct_labels, output_layer_activations))
        for cost_index, cost in enumerate(costs[0]):
            costs[0][cost_index] = cost*self.sigmoid_prime(output_layer_outputs[cost_index])

        # calculate costs for layers
        for layer_index, layer_size in enumerate(self.sizes[::-1][1:-1], start=1):
            layer_costs = []
            layer_weights = self.weights[-layer_index]
            layer_outputs = self.outputs[-(layer_index+1)]
            previous_layer_costs = costs[layer_index-1]

            next_layer_size = self.sizes[::-1][1:][layer_index]

            layer_weights_formatted = []
            for x in range(layer_size):
                layer_weights_formatted.append([])
            for weight_index, weight in enumerate(layer_weights, start=0):
                #print(f"weight index:{weight_index % next_layer_size} layer_index: {weight_index}")
                layer_weights_formatted[weight_index%layer_size].append(layer_weights[weight_index])

            #print(f"next_layer_size:{layer_size} costs: {len(previous_layer_costs)}, layer_weights_formatted: {layer_weights_formatted}")
            for x in range(layer_size):
                node_cost = 0
                for y, cost in enumerate(previous_layer_costs,start=0):
                    node_cost += layer_weights_formatted[x][y]*cost
                layer_costs.append(node_cost)

            # layer_costs same order as next layer's activations
            for cost_index, cost in enumerate(layer_costs):
                layer_costs[cost_index] = cost * self.sigmoid_prime(layer_outputs[cost_index])

            costs.append(layer_costs)
        # calculate weight errors
        weight_errors = []
        bias_errors = []
        for layer_index, layer_costs in enumerate(costs[::-1]):
            layer_activations = self.activations[layer_index]
            layer_weight_errors = []
            for cost_index, cost in enumerate(layer_costs,start=0):
                for activation in layer_activations:
                    layer_weight_errors.append(activation * cost)
            weight_errors.append(np.array(layer_weight_errors))
            bias_errors.append(sum(layer_costs))

        return weight_errors, bias_errors

    # conversion tool
    def translate_label_to_array(self, y):
        translated_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        translated_label[y] = 1
        return np.array(translated_label)

    # output tools

    def network_outputs(self):
        print("Output layer: ")
        for x in range(self.sizes[-1]):
            print("node " + str(x) + ": " + str(self.activations[-1][x]))

    def total_activations(self):
        print(len(self.activations))

    def compute_cost_derivative(self, y, output_activations):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    def sigmoid(self, z):
        """"The sigmoid function."""
        return (1.0 / (1.0 + np.exp(-z)))

    def sigmoid_prime(self, z):
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))