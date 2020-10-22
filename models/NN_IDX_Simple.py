
import idx2numpy
import numpy
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import scipy.special
from tqdm import trange

class neuralNetwork:
    # initialise the neural network
    def __init__(self, ipnodes, h1nodes, h2nodes, opnodes, bias_hid2op, bias_hid1hid2, bias_iphid1, momentum, learningrate):
# set number of nodes in each input, hidden, output layer
        self.ip_nodes = ipnodes
        self.h1_nodes = h1nodes
        self.h2_nodes = h2nodes
        self.op_nodes = opnodes

        self.bias_h2op = bias_hid2op
        self.bias_h1h2 = bias_hid1hid2
        self.bias_iph1 = bias_iphid1
        #Momentum
        self.beta = momentum
        self.Vdw_h2_op = 0
        self.Vdw_h1_h2 = 0
        self.Vdw_ip_h1 = 0
        
        self.Vdb_h2_op = 0
        self.Vdb_h1_h2 = 0
        self.Vdb_ip_h1 = 0
        #Linking Biases
        self.bias_h2_op = numpy.random.randn(self.bias_h2op, 1)
        self.bias_h1_h2 = numpy.random.randn(self.bias_h1h2, 1)
        self.bias_ip_h1 = numpy.random.randn(self.bias_iph1, 1)
        # Linking weights 
        self.w_ip_h1 = numpy.random.randn(self.h1_nodes, self.ip_nodes)
        self.w_h1_h2 = numpy.random.randn(self.h2_nodes, self.h1_nodes)
        self.w_h2_op = numpy.random.randn(self.op_nodes, self.h2_nodes)
        
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    '''
    def activation_function(self,x):
        self.x = (1 / (1 + numpy.e**(-x)))
        return self.x'''
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
        targets = numpy.array(targets_list, ndmin=2)
        
        X_h1 = numpy.dot(self.w_ip_h1, inputs) 
        O_h1 = self.activation_function(X_h1)
        # calculate signals into hidden layer
        X_h2 = numpy.dot(self.w_h1_h2, O_h1) 
        O_h2 = self.activation_function(X_h2)
        # calculate signals into final output layer
        X_op = numpy.dot(self.w_h2_op, O_h2)
        O_op = self.activation_function(X_op)
        # output layer error is the (target - actual)
        errors_op = targets - O_op
        errors_h2 = numpy.dot(self.w_h2_op.T, errors_op)
        errors_h1 = numpy.dot(self.w_h1_h2.T, errors_h2)
        
        self.dw_h2_op = numpy.dot((errors_op * O_op * (1.0 - O_op)), numpy.transpose(O_h2))
        self.dw_h1_h2 = numpy.dot((errors_h2 * O_h2 * (1.0 - O_h2)), numpy.transpose(O_h1))
        self.dw_ip_h1 = numpy.dot((errors_h1 * O_h1 * (1.0 - O_h1)), numpy.transpose(inputs))
        
        self.Vdw_h2_op =  beta*self.Vdw_h2_op +(1-beta)*self.dw_h2_op
        self.Vdw_h1_h2 =  beta*self.Vdw_h1_h2 +(1-beta)*self.dw_h1_h2
        self.Vdw_ip_h1 =  beta*self.Vdw_ip_h1 +(1-beta)*self.dw_ip_h1
        
        # update the weights for the links between the hidden and output layers
        self.w_h2_op += self.lr * self.Vdw_h2_op
        self.w_h1_h2 += self.lr * self.Vdw_h1_h2
        self.w_ip_h1 += self.lr * self.Vdw_ip_h1
        return O_op
# query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into 1st hidden layer
        X_h1 = numpy.dot(self.w_ip_h1, inputs) 
        O_h1 = self.activation_function(X_h1)
        # calculate signals into 2nd hidden layer
        X_h2 = numpy.dot(self.w_h1_h2, O_h1) 
        O_h2 = self.activation_function(X_h2)
        # calculate signals into final output layer
        X_op = numpy.dot(self.w_h2_op, O_h2) 
        O_op = self.activation_function(X_op)
        return O_op
#%%     
# number of input, hidden and output nodes
input_nodes = 784
nodes_in_1st_hidden_layer = 100
nodes_in_2nd_hidden_layer = 100
output_nodes = 10
# learning rate
learning_rate = 0.0001
#Momentum
beta = 0.7
#Data Aranging for NN Class
hidden1_nodes = nodes_in_1st_hidden_layer
hidden2_nodes = nodes_in_2nd_hidden_layer
bias_iph1 = nodes_in_1st_hidden_layer
bias_h1h2 = nodes_in_2nd_hidden_layer
bias_h2op = output_nodes
# create instance of neural network
n = neuralNetwork(input_nodes,hidden1_nodes,hidden2_nodes, output_nodes, bias_h2op, bias_h1h2, bias_iph1, beta, learning_rate)
#%%
#------------------------------------------------------------------------Loading Training Data----------------------------------------------------------
training_data_ip = idx2numpy.convert_from_file('train-images.idx3-ubyte')
training_data_label = idx2numpy.convert_from_file('train-labels.idx1-ubyte')

# normalizing the data between [0, 1] (i.e. not exactly 1) 
inputs_train = ((training_data_ip / 255.0) * 0.99) + 0.01
inputs_train = numpy.reshape(inputs_train, (60000,784))
    
targets_train = numpy.zeros([10,60000]) + 0.01
for c in range(len(training_data_label)):
   r = training_data_label[c]
   targets_train[r][c] = 0.99 
pass

#%%
#---------------------------------------------------------------------Loading Testing Data-------------------------------------------------------------
test_data_ip = idx2numpy.convert_from_file('t10k-images.idx3-ubyte.idx3-ubyte')
test_data_label = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


inputs_test = ((test_data_ip / 255.0) * 0.99) + 0.01
inputs_test = numpy.reshape(inputs_test, (10000,784))

targets_test = numpy.zeros([10,10000]) + 0.01
for c in range(len(test_data_label)):
       r = test_data_label[c]
       targets_test[r][c] = 0.99 
       
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Check Data Loading')
ax1.imshow(training_data_ip[0,:,:], cmap='gray')
ax1.set_title('Training image')
ax2.imshow(test_data_ip[0,:,:], cmap='gray')
ax2.set_title('Test image')
#%%
#---------------------------------------------------------------Training NN---------------------------------------------------------------------------
Accuracy = []
cost = []
epochs = 4
plot_epoch = epochs
go1 = time.time()
for e in trange(epochs, desc = 'Training Neural Network: Epoch Done'):
    # creating batches
    correct=0
    inputs_train_batch, targets_train_bacth = [] , []
    Batch_size = 10000
    BS_loop = Batch_size
    Batches = 60000 / Batch_size 
    Batches = int(Batches)
    
    start = 0
    go2 = time.time()
    for i in range(Batches):
        #O_train = n.train(inputs_train_batch, targets_train_bacth)
        inputs_train_batch = inputs_train[start:Batch_size, :]
        targets_train_bacth = targets_train[:, start:Batch_size]
        # calling train function of neural network
        O_train = n.train(inputs_train_batch, targets_train_bacth)
        start = Batch_size
        Batch_size = Batch_size + BS_loop
        #Cost Calculate
        Average_error = numpy.sum(O_train,axis=0) / 10
        Cost_func =  (1/ (2 * BS_loop)) * (sum(Average_error)**2)
        #-----------Testing on Test Data Set Each Epoch----------------------
        outputs = n.query(inputs_test)
        label = numpy.argmax(outputs, axis=0)
        correct = 0
        for j in range(10000):
            if (test_data_label [j] == label[j]):
                correct+=1
        Performance = correct / 10000
    pass
    cost.append(Cost_func)
    Accuracy.append(Performance)
    end2 = time.time() - go2
pass
end1 = time.time() - go1
#%%
#---------------------------------------------------------Optimizayion Algorithm & Printing----------------------------------------------------------
print('='*40)
if BS_loop == 60000:
  print("The Optimization Algorithm was Batch G.D")
elif BS_loop == 1:
  print("The Optimization Algorithm was Stochastic G.D")
else:
  print("The Optimization Algorithm was Mini_Batch G.D")
  
print("Time taken for 1 Epoch=",end2,'Seconds')
print("Total time taken=",end1,'Seconds')
print ("Learning Rate=", learning_rate, "\nMomentum Term=", beta, "\nBatch Size=", BS_loop,\
       "\nEpoch=", epochs, "\nPerformance=", Performance * 100,'%')
print('='*40)
#%%
#-------------------------------------------------------------Plotting------------------------------------------------------------------------------

plot = plot_epoch
#Learning Curve
p = numpy.linspace(1,plot,plot)
fig = plt.figure(figsize=(8,8))
plt.plot(p, cost, 'b')
plt.xlabel('No. of Epochs')
plt.ylabel('Cost Function')
plt.show()


#Performance Curve
p=numpy.linspace(1,plot,plot)
fig = plt.figure(figsize=(8,8))
plt.plot(p, Accuracy, 'r')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.show()






