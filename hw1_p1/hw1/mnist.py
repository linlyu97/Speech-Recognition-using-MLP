"""Problem 3 - Training on MNIST"""
import math
import numpy as np

from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.optim.optimizer import Optimizer
from mytorch.optim.sgd import SGD
from mytorch.nn.linear import Linear
from mytorch.nn.activations import ReLU
from mytorch.nn.sequential import Sequential
from mytorch.nn.loss import CrossEntropyLoss
from random import shuffle


# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Problem 3.1: Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    
    model = Sequential(Linear(784,20), ReLU(), Linear(20,10))
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0)
    val_accuracies=train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3)
    # TODO: Call training routine (make sure to write it below)
    
    return val_accuracies

def get_batches(train_x,train_y):
    batches=[]
    num_batch=math.ceil(len(train_x)/BATCH_SIZE)
    for i in range(num_batch):
        batches.append((train_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
        train_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))
    return batches

def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Problem 3.2: Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    #model = sequential.Sequential(Linear(784,20), ReLU(), Linear(20,10))
    #print('---')
    val_accuracies = []
    for epoch in range(num_epochs):
        list1_shuf = []
        list2_shuf = []
        index_shuf = list(range(len(train_x)))
        shuffle(index_shuf)
        #print(len(index_shuf))
        for i in index_shuf:
            #print('i',i)
            list1_shuf.append(train_x[i])
            list2_shuf.append(train_y[i])
        array1=np.stack(list1_shuf)
        array2=np.stack(list2_shuf)
        batches=get_batches(array1,array2)
        #print('batches:',batches)
        #print(batches.shape)
        for i, tuple1 in enumerate(batches):
            batch_data, batch_labels=tuple1
            optimizer.zero_grad()
            batch_data=Tensor(batch_data)
            batch_labels=Tensor(batch_labels)
            out = model.forward(batch_data)
            #print(out)
            loss = criterion.forward(out, batch_labels)
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                #print(accuracy)
                model.train()
    # TODO: Implement me! (Pseudocode on writeup)
    return val_accuracies

def validate(model, val_x, val_y):
    """Problem 3.3: Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    #TODO: implement validation based on pseudocode
    model.eval()
    batches=get_batches(val_x,val_y)
    num_correct=0
    for (batch_data, batch_labels) in batches:
        batch_data=Tensor(batch_data)
        out = model.forward(batch_data)
        batch_preds = np.argmax(out.data, axis=1)
        #print('ba,pre',batch_preds)
        #print('ba_true',batch_labels)
        for i in range(len(batch_preds)):
            if batch_preds[i] == batch_labels[i]:
                num_correct+=1
    #print(num_correct)
    accuracy = num_correct / len(val_x)
    return accuracy
