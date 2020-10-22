# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 14:31:20 2019

@author: Talha Ilyas
"""
import numpy

class neuralNetwork:
    # initialise the neural network
    def __init__(self, ipnodes, h1nodes, h2nodes, opnodes, bias_hid2op, bias_hid1hid2, bias_iphid1, learningrate):
# set number of nodes in each input, hidden, output layer
        self.ip_nodes = ipnodes
        self.h1_nodes = h1nodes
        self.h2_nodes = h2nodes
        self.op_nodes = opnodes

        self.bias_h2op = bias_hid2op
        self.bias_h1h2 = bias_hid1hid2
        self.bias_iph1 = bias_iphid1
        
        #Linking Biases
        self.bias_h2_op = numpy.random.randn(self.bias_h2op, 1)
        self.bias_h1_h2 = numpy.random.randn(self.bias_h1h2, 1)
        self.bias_ip_h1 = numpy.random.randn(self.bias_iph1, 1)
        # Linking weights 
        self.w_ip_h1 = numpy.random.normal(0.0, pow(self.h1_nodes, -0.5),(self.h1_nodes, self.ip_nodes))
        self.w_h1_h2 = numpy.random.normal(0.0, pow(self.h2_nodes, -0.5),(self.h2_nodes, self.h1_nodes))
        self.w_h2_op = numpy.random.normal(0.0, pow(self.op_nodes, -0.5),(self.op_nodes, self.h2_nodes))
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        #self.sigmoid_function = lambda x: (1 / (1 + numpy.e**(-x)))
        pass
    def sigmoid_function(self,x):
        self.x = (1 / (1 + numpy.e**(-x)))
        return self.x
    def ReLu(self,x):
        self.x = numpy.where(x<=0, 0, x)
        return self.x
    def d_ReLu(self,x):
        self.x = numpy.where(x<=0, 0, 1)
        return self.x
# train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        X_h1 = numpy.add(numpy.dot(self.w_ip_h1, inputs) , self.bias_ip_h1)
        O_h1 = self.ReLu(X_h1)
        # calculate signals into hidden layer
        X_h2 = numpy.add(numpy.dot(self.w_h1_h2, O_h1) , self.bias_h1_h2)
        O_h2 = self.ReLu(X_h2)
        # calculate signals into final output layer
        X_op = numpy.add(numpy.dot(self.w_h2_op, O_h2) , self.bias_h2_op)
        #O_op = self.ReLu(X_op)
        O_op = self.sigmoid_function(X_op)#For Sigmoid at output layer
        
        error_op = targets - O_op 
        errors_op = (targets - O_op) * O_op * (1.0 - O_op)#For Sigmoid at output layer
        #errors_op = (targets - O_op) * self.d_ReLu(X_op)#For ReLu at output layer
        errors_h2 = numpy.dot(self.w_h2_op.T, errors_op)
        errors_h1 = numpy.dot(self.w_h1_h2.T, errors_h2)
        
        # update the weights for the links between the hidden and output layers
        self.w_h2_op += self.lr * numpy.dot((error_op * O_op * (1.0 - O_op)), numpy.transpose(O_h2))#For Sigmoid at output layer
        #self.w_h2_op += self.lr * numpy.dot((error_op *  self.d_ReLu(O_op)), numpy.transpose(O_h2))#For ReLu at output layer
        self.w_h1_h2 += self.lr * numpy.dot((errors_h2 * self.d_ReLu(X_h2)), numpy.transpose(O_h1))
        self.w_ip_h1 += self.lr * numpy.dot((errors_h1 * self.d_ReLu(X_h1)), numpy.transpose(inputs))
        # update the biases for the links between the hidden and output layers
        self.bias_h2_op += self.lr * (errors_op * O_op * (1.0 - O_op))#For Sigmoid at output layer
        self.bias_h1_h2 += self.lr * (errors_h2 * self.d_ReLu(X_h2))
        self.bias_ip_h1 += self.lr * (errors_h1 * self.d_ReLu(X_h1))
        #self.bias_h2_op += self.lr * (errors_op * self.d_ReLu(O_op))#For ReLu at output layer
        return O_op
# query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into 1st hidden layer
        X_h1 = numpy.add(numpy.dot(self.w_ip_h1, inputs) , self.bias_ip_h1)
        O_h1 = self.ReLu(X_h1)
        # calculate signals into 2nd hidden layer
        X_h2 = numpy.add(numpy.dot(self.w_h1_h2, O_h1) , self.bias_h1_h2)
        O_h2 = self.ReLu(X_h2)
        # calculate signals into final output layer
        X_op = numpy.add(numpy.dot(self.w_h2_op, O_h2) , self.bias_h2_op)
        O_op = self.sigmoid_function(X_op)#For Sigmoid at output layer
        #O_op = self.d_ReLu(X_op)#For ReLu at output layer
        return O_op
       
# number of input, hidden and output nodes
input_nodes = 784
nodes_in_1st_hidden_layer = 100
nodes_in_2nd_hidden_layer = 100
output_nodes = 10
# learning rate
learning_rate = 0.1
#Momentum
beta = 0.9
#Data Aranging for NN Class
hidden1_nodes = nodes_in_1st_hidden_layer
hidden2_nodes = nodes_in_2nd_hidden_layer
bias_iph1 = nodes_in_1st_hidden_layer
bias_h1h2 = nodes_in_2nd_hidden_layer
bias_h2op = output_nodes
# create instance of neural network
n = neuralNetwork(input_nodes,hidden1_nodes,hidden2_nodes, output_nodes, bias_h2op, bias_h1h2, bias_iph1, learning_rate)
# load the mnist training data CSV file into a list
training_data_file = open("E:\Anaconda\Data CSV\mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# train the neural network
# epochs is the number of times the training data set is used for training
epochs = 1
for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        cost=0
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass
pass

# load the mnist test data CSV file into a list
test_data_file = open("E:\Anaconda\Data CSV\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to
        scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to
        scorecard
        scorecard.append(0)
    pass
pass
# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() /scorecard_array.size)