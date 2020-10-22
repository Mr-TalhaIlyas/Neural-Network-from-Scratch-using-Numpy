# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:23:24 2019

@author: Talha Ilyas
"""

import idx2numpy
import numpy
import time
import matplotlib.pyplot as plt
import scipy.special
from tqdm import trange
class neuralNetwork:
    # initialise the neural network
    '''
    Creates a class named neural network with 3 methods
    __init__: for initilizing the class
    train   : call this method for training neural network made with given specs
    query   : call this method for testing neural network
    Parameters
    ----------
    ipnodes : number of nodes in the 1st layer i.e. input layer (in case of MNIST => 28x28=728)
    h1nodes : number of nodes in the 2nd layer i.e. 1st hidden layer (greater the number heavier the network)
    h2nodes : number of nodes in the 3rd layer i.e. 2nd hidden layer (greater the number heavier the network)
    opnodes : number of nodes in the 4th layer i.e. output hidden layer (equal to number of classes i.e. 10)
    bias_hid2op : Biases matrix between 2nd hidden layer and output 
    bias_hid1hid2 : Biases matrix between 1st hidden layer and 2nd hidden layer
    bias_iphid1 : Biases matrix between input layer and 1st hidden layer
    Epsilion  : a small float const to avoid division by zero
    BatchSize : a int constant for to select between BGD, MBGD and SGD
    momentum_1 : a float constant for optimizer, (beta 1) of ADAM
    momentum_2 : a float constant for optimizer, (beta 2) of ADAM
    learningrate : a float constantto take step in gradine descent direction
    
    Returns
    -------
    train   : returns ouptuts of network after each Epoch for performance measure
    query   : returns predictions of network for a given input

    '''
    def __init__(self, ipnodes, h1nodes, h2nodes, opnodes, bias_hid2op, bias_hid1hid2, bias_iphid1, momentum_1, momentum_2, Epsilion, BatchSize, learningrate):
        
# set number of nodes in each input, hidden, output layer
        self.ip_nodes = ipnodes
        self.h1_nodes = h1nodes
        self.h2_nodes = h2nodes
        self.op_nodes = opnodes

        self.bias_h2op = bias_hid2op
        self.bias_h1h2 = bias_hid1hid2
        self.bias_iph1 = bias_iphid1
        
        self.batch_size = BatchSize
        #1st and second Moments
        self.beta = momentum_1
        self.gamma = momentum_2
        self.epsilion = Epsilion
        #1st Moments Weights
        self.Vdw_h2_op = 0
        self.Vdw_h1_h2 = 0
        self.Vdw_ip_h1 = 0
        #1st Moments Biases
        self.Vdb_h2_op = 0
        self.Vdb_h1_h2 = 0
        self.Vdb_ip_h1 = 0
        #@nd Moments Weights
        self.Sdw_h2_op = 0
        self.Sdw_h1_h2 = 0
        self.Sdw_ip_h1 = 0
        #2nd Moments Biases
        self.Sdb_h2_op = 0
        self.Sdb_h1_h2 = 0
        self.Sdb_ip_h1 = 0
         #Linking Biases
        #Guassian Normal Distribution pow() means deviation in values is between +- h2_nodes**-0.5 with mean=0
        self.bias_h2_op = numpy.random.normal(0.0, pow(self.bias_h2op, -0.5),(self.bias_h2op, 1))
        self.bias_h1_h2 = numpy.random.normal(0.0, pow(self.bias_h1h2, -0.5),(self.bias_h1h2, 1))
        self.bias_ip_h1 = numpy.random.normal(0.0, pow(self.bias_iph1, -0.5),(self.bias_iph1, 1))
        # Linking weights
        #Guassian Normal Distribution pow() means deviation in values is between +- h2_nodes**-0.5 with mean=0
        self.w_ip_h1 = numpy.random.normal(0.0, pow(self.h1_nodes, -0.5),(self.h1_nodes, self.ip_nodes))
        self.w_h1_h2 = numpy.random.normal(0.0, pow(self.h2_nodes, -0.5),(self.h2_nodes, self.h1_nodes))
        self.w_h2_op = numpy.random.normal(0.0, pow(self.op_nodes, -0.5),(self.op_nodes, self.h2_nodes))
        
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    def ReLu(self,x):
        self.x = numpy.where(x<=0, 0, x)
        return self.x
    def d_ReLu(self,x):
        self.x = numpy.where(x<=0, 0, 1)
        return self.x
# train the neural network
    def train(self, inputs_list, targets_list, iteration):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2)
        t = iteration + 1 #Because Python will start count form 0 
        
        X_h1 = numpy.add(numpy.dot(self.w_ip_h1, inputs) , self.bias_ip_h1)
        O_h1 = self.activation_function(X_h1)
        # calculate signals into hidden layer
        X_h2 = numpy.add(numpy.dot(self.w_h1_h2, O_h1) , self.bias_h1_h2)
        O_h2 = self.activation_function(X_h2)
        # calculate signals into final output layer
        X_op = numpy.add(numpy.dot(self.w_h2_op, O_h2) , self.bias_h2_op)
        O_op = self.activation_function(X_op)
        # output layer error is the (target - actual)
        errors_op = targets - O_op
        errors_h2 = numpy.dot(self.w_h2_op.T, errors_op)
        errors_h1 = numpy.dot(self.w_h1_h2.T, errors_h2)
        #Weights and Biases Gradients
        self.dw_h2_op = numpy.dot((errors_op * O_op * (1.0 - O_op)), numpy.transpose(O_h2))
        self.dw_h1_h2 = numpy.dot((errors_h2 * O_h2 * (1.0 - O_h2)), numpy.transpose(O_h1))
        self.dw_ip_h1 = numpy.dot((errors_h1 * O_h1 * (1.0 - O_h1)), numpy.transpose(inputs))
        
        self.db_h2_op = (numpy.sum(errors_op *O_op * (1.0 - O_op))) / self.batch_size
        self.db_h1_h2 = (numpy.sum(errors_h2 *O_h2 * (1.0 - O_h2))) / self.batch_size
        self.db_ip_h1 = (numpy.sum(errors_h1 *O_h1 * (1.0 - O_h1))) / self.batch_size
        #1st and 2nd Moment Weights
        self.Vdw_h2_op =  beta*self.Vdw_h2_op +(1-beta) * self.dw_h2_op
        self.Vdw_h1_h2 =  beta*self.Vdw_h1_h2 +(1-beta) * self.dw_h1_h2
        self.Vdw_ip_h1 =  beta*self.Vdw_ip_h1 +(1-beta) * self.dw_ip_h1
        #2nd
        self.Sdw_h2_op =  gamma*self.Sdw_h2_op +(1-gamma) * numpy.square(self.dw_h2_op)
        self.Sdw_h1_h2 =  gamma*self.Sdw_h1_h2 +(1-gamma) * numpy.square(self.dw_h1_h2)
        self.Sdw_ip_h1 =  gamma*self.Sdw_ip_h1 +(1-gamma) * numpy.square(self.dw_ip_h1)
        #1st and 2nd Moment Biases
        self.Vdb_h2_op =  beta*self.Vdb_h2_op +(1-beta)*self.db_h2_op
        self.Vdb_h1_h2 =  beta*self.Vdb_h1_h2 +(1-beta)*self.db_h1_h2
        self.Vdb_ip_h1 =  beta*self.Vdb_ip_h1 +(1-beta)*self.db_ip_h1
        #2nd
        self.Sdb_h2_op =  gamma*self.Sdb_h2_op +(1-gamma)*numpy.square(self.db_h2_op)
        self.Sdb_h1_h2 =  gamma*self.Sdb_h1_h2 +(1-gamma)*numpy.square(self.db_h1_h2)
        self.Sdb_ip_h1 =  gamma*self.Sdb_ip_h1 +(1-gamma)*numpy.square(self.db_ip_h1)
        #1st and 2nd Moment corrected Weights
        Vdw_h2_op_cor =  self.Vdw_h2_op / (1-beta**t)
        Vdw_h1_h2_cor =  self.Vdw_h1_h2 / (1-beta**t)
        Vdw_ip_h1_cor =  self.Vdw_ip_h1 / (1-beta**t)
        #2nd
        Sdw_h2_op_cor =  self.Sdw_h2_op /(1-gamma**t) 
        Sdw_h1_h2_cor =  self.Sdw_h1_h2 /(1-gamma**t) 
        Sdw_ip_h1_cor =  self.Sdw_ip_h1 /(1-gamma**t)
        #1st and 2nd Moment corrected Biases
        Vdb_h2_op_cor =  self.Vdb_h2_op /(1-beta**t)
        Vdb_h1_h2_cor =  self.Vdb_h1_h2 /(1-beta**t)
        Vdb_ip_h1_cor =  self.Vdb_ip_h1 /(1-beta**t)
        #2nd
        Sdb_h2_op_cor =  self.Sdb_h2_op /(1-gamma**t)
        Sdb_h1_h2_cor =  self.Sdb_h1_h2 /(1-gamma**t)
        Sdb_ip_h1_cor =  self.Sdb_ip_h1 /(1-gamma**t)
        # update the weights for the links between the hidden and output layers
        self.w_h2_op += self.lr * Vdw_h2_op_cor/(numpy.sqrt(Sdw_h2_op_cor)+epsilion)
        self.w_h1_h2 += self.lr * Vdw_h1_h2_cor/(numpy.sqrt(Sdw_h1_h2_cor)+epsilion)
        self.w_ip_h1 += self.lr * Vdw_ip_h1_cor/(numpy.sqrt(Sdw_ip_h1_cor)+epsilion)
        
        # update the biases for the links between the hidden and output layers
        self.bias_h2_op += self.lr * Vdb_h2_op_cor/(numpy.sqrt(Sdb_h2_op_cor)+epsilion)
        self.bias_h1_h2 += self.lr * Vdb_h1_h2_cor/(numpy.sqrt(Sdb_h1_h2_cor)+epsilion)
        self.bias_ip_h1 += self.lr * Vdb_ip_h1_cor/(numpy.sqrt(Sdb_ip_h1_cor)+epsilion)
        return errors_op
# query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into 1st hidden layer
        X_h1 = numpy.add(numpy.dot(self.w_ip_h1, inputs) , self.bias_ip_h1)
        O_h1 = self.activation_function(X_h1)
        # calculate signals into 2nd hidden layer
        X_h2 = numpy.add(numpy.dot(self.w_h1_h2, O_h1) , self.bias_h1_h2)
        O_h2 = self.activation_function(X_h2)
        # calculate signals into final output layer
        X_op = numpy.add(numpy.dot(self.w_h2_op, O_h2) , self.bias_h2_op)
        O_op = self.activation_function(X_op)
        return O_op
#%%     
# number of input, hidden and output nodes
input_nodes = 784
nodes_in_1st_hidden_layer = 100
nodes_in_2nd_hidden_layer = 100
output_nodes = 10
# learning rate
learning_rate = 0.001
#ADAM 1st and 2nd Moments
beta = 0.9
gamma = 0.999
epsilion = 1e-8
#For scaling Bias Updates
Global_Batchsize = 1
#Epochs -or- iteration
epochs = 3
#Data Aranging for NN Class
hidden1_nodes = nodes_in_1st_hidden_layer
hidden2_nodes = nodes_in_2nd_hidden_layer
bias_iph1 = nodes_in_1st_hidden_layer
bias_h1h2 = nodes_in_2nd_hidden_layer
bias_h2op = output_nodes
# create instance of neural network
n = neuralNetwork(input_nodes,hidden1_nodes,hidden2_nodes, output_nodes, bias_h2op, bias_h1h2, bias_iph1, beta, gamma, epsilion, Global_Batchsize, learning_rate)
#%%
#------------------------------------------------------------------------Loading Training Data----------------------------------------------------------
training_data_ip = idx2numpy.convert_from_file('mnist_data/train-images.idx3-ubyte')
training_data_label = idx2numpy.convert_from_file('mnist_data/train-labels.idx1-ubyte')

inputs_train = ((training_data_ip / 255.0) * 0.99) + 0.01
inputs_train = numpy.reshape(inputs_train, (60000,784))
    
targets_train = numpy.zeros([10,60000]) + 0.01
for c in range(len(training_data_label)):
   r = training_data_label[c]
   targets_train[r][c] = 0.99 
pass
#---------------------------------------------------------------------Loading Testing Data--------------------------------------------------------------------------
test_data_ip = idx2numpy.convert_from_file('mnist_data/t10k-images.idx3-ubyte.idx3-ubyte')
test_data_label = idx2numpy.convert_from_file('mnist_data/t10k-labels.idx1-ubyte')


inputs_test = ((test_data_ip / 255.0) * 0.99) + 0.01
inputs_test = numpy.reshape(inputs_test, (10000,784))

targets_test = numpy.zeros([10,10000]) + 0.01
for c1 in range(len(test_data_label)):
       r1 = test_data_label[c1]
       targets_test[r1][c1] = 0.99

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Check Data Loading')
ax1.imshow(training_data_ip[0,:,:], cmap='gray')
ax1.set_title('Training image')
ax2.imshow(test_data_ip[0,:,:], cmap='gray')
ax2.set_title('Test image')
#%%
#---------------------------------------------------------------Training NN-----------------------------------------------------------------------------

cost = []
plot_epoch = epochs
go1 = time.time()
for e in trange(epochs, desc = 'Training Neural Network: Epoch Done'):
    
    inputs_train_batch, targets_train_bacth = [] , []
    Batch_size = Global_Batchsize
    BS_loop = Batch_size
    Batches = 60000 / Batch_size 
    Batches = int(Batches)
    
    start = 0
    go2 = time.time()
    for i in range(Batches):
        inputs_train_batch = inputs_train[start:Batch_size, :]
        targets_train_bacth = targets_train[:, start:Batch_size]
        Errors_train = n.train(inputs_train_batch, targets_train_bacth, i)
        start = Batch_size
        Batch_size = Batch_size + BS_loop
        #Cost Calculate
        Average_error = numpy.sum(Errors_train,axis=0) / 10
        Cost_func =  (1/ (2 * BS_loop)) * (sum(Average_error**2))
        pass
    cost.append(Cost_func)
    end2 = time.time() - go2
pass
end1 = time.time() - go1
#%%
#----------------------------------------------------------------------Performance Measure-------------------------------------------------------------
outputs = n.query(inputs_test)
correct=0
label = numpy.argmax(outputs, axis=0)
Accuracy = []
for i in range(10000):
    if (test_data_label [i]==label[i] ):
        # network's answer matches correct answer, add 1 to
       correct+=1
Performance = correct / 10000
#---------------------------------------------------------Optimizayion Algorithm & Printing----------------------------------------------------------
print('='*40)
if BS_loop == 60000:
  print("The Optimization Algorithm was Batch G.D with ADAM")
elif BS_loop == 1:
  print("The Optimization Algorithm was Stochastic G.D with ADAM")
else:
  print("The Optimization Algorithm was Mini_Batch G.D with ADAM")
  
print("Time taken for 1 Epoch=",end2,'Seconds')
print("Total time taken=",end1,'Seconds')
print ("Learning Rate=", learning_rate, "1st Moment(beta)=", beta,"2nd Moment(gamma)=",gamma, "Epsilion=",epsilion)
print("Batch Size=", BS_loop, "Epoch=", epochs, "Performance=", Performance * 100,'%')
print('='*40)
#-------------------------------------------------------------Plotting------------------------------------------------------------------------------

plot = plot_epoch
#Learning Curve
p = numpy.linspace(1,plot_epoch,plot_epoch)
fig = plt.figure(figsize=(8,8))
plt.plot(p, cost, 'b')
plt.xlabel('No. of Epochs')
plt.ylabel('Cost Function')
plt.show()

