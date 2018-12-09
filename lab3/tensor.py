'''

tensor.py, code was adapted from Vivian Rajkumar:

- https://medium.com/tensorist/classifying-fashion-articles-using-tensorflow-fashion-mnist-f22e8a04728a

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

def viz_tensor(instances, fashion_mnist, labels):
    '''

    Visualize corresponding elements.

    '''

    plt.figure(figsize=(10,10))
    for i, instance in enumerate(instances):
        # sample: get 28x28 image
        sample = fashion_mnist.train.images[instance].reshape(28,28)

        # integer label from one-hot encoded data
        sample_label = np.where(fashion_mnist.train.labels[instance] == 1)[0][0]

        # plot sample
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample, cmap='Greys')
        plt.xlabel('index = {index} ({label})'.format(
            index=sample_label,
            label=labels[sample_label]
        ))

def create_placeholders(n_x, n_y):
    '''

    Creates the placeholders for the tensorflow session.
 
    Arguments:
    n_x, scalar, size of an image vector (28*28 = 784)
    n_y, scalar, number of classes (10)
 
    Returns:
    X, placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y, placeholder for the input labels, of shape [n_y, None] and dtype "float"

    '''
 
    X = tf.placeholder(tf.float32, [n_x, None], name='X')
    Y = tf.placeholder(tf.float32, [n_y, None], name='Y')
 
    return X, Y

def initialize_parameters(hidden_1, hidden_2, n_input, n_classes):
    '''

    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [hidden_1, n_input]
                        b1 : [hidden_1, 1]
                        W2 : [hidden_2, hidden_1]
                        b2 : [hidden_2, 1]
                        W3 : [n_classes, hidden_2]
                        b3 : [n_classes, 1]
    
    Returns:
    parameters, a dictionary of tensors containing W1, b1, W2, b2, W3, b3

    '''
    
    # random seed
    tf.set_random_seed(11)

    #
    # Initialize weights and biases for each layer
    #

    # first hidden layer
    W1 = tf.get_variable(
        'W1',
        [hidden_1, n_input],
        initializer=tf.contrib.layers.xavier_initializer(seed=11)
    )
    b1 = tf.get_variable('b1',
        [hidden_1, 1],
        initializer=tf.zeros_initializer()
    )
    
    # second hidden layer
    W2 = tf.get_variable(
        'W2',
        [hidden_2, hidden_1],
        initializer=tf.contrib.layers.xavier_initializer(seed=11)
    )
    b2 = tf.get_variable(
        'b2',
        [hidden_2, 1],
        initializer=tf.zeros_initializer()
    )
    
    # output layer
    W3 = tf.get_variable(
        'W3',
        [n_classes, hidden_2],
        initializer=tf.contrib.layers.xavier_initializer(seed=42)
    )
    b3 = tf.get_variable(
        'b3',
        [n_classes, 1],
        initializer=tf.zeros_initializer()
    )
    
    # store initializations as a dictionary of parameters
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    
    return parameters

def forward_propagation(X, parameters):
    '''

    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X, input dataset placeholder, of shape (input size, number of examples)
    parameters, python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3, the output of the last LINEAR unit

    '''
    
    # Retrieve parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    # Carry out forward propagation      
    Z1 = tf.add(tf.matmul(W1,X), b1)     
    A1 = tf.nn.relu(Z1)                  
    Z2 = tf.add(tf.matmul(W2,A1), b2)    
    A2 = tf.nn.relu(Z2)                  
    Z3 = tf.add(tf.matmul(W3,A2), b3)    
    
    return Z3

def compute_cost(Z3, Y, labels):
    '''

    Computes the cost
    
    Arguments:
    Z3, output of forward propagation (output of the last LINEAR unit), of shape (10, number_of_examples)
    Y, "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function

    '''
    
    # get logits (predictions) and labels
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    ))
    
    return cost

def model(
    train,
    test,
    labels,
    learning_rate=0.0001,
    num_epochs=16,
    minibatch_size=32,
    print_cost=True,
    hidden_1=128,
    hidden_2=128,
    n_input=784,
    n_classes=10
):
    '''

    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    train, training set
    test, test set
    learning_rate, learning rate of the optimization
    num_epochs, number of epochs of the optimization loop
    minibatch_size, size of a minibatch
    print_cost, True to print the cost every epoch
    hidden_1, units in first hidden layer
    hidden_2, units in second hidden layer
    n_input, mnist data input (hidden_1 x hidden_2)
    n_classes, mnist total classes
    
    Returns:
    parameters, parameters learnt by the model. They can then be used to predict.

    '''
    
    # Ensure that model can be rerun without overwriting tf variables
    ops.reset_default_graph()
    # For reproducibility
    tf.set_random_seed(42)
    seed = 42
    # Get input and output shapes
    (n_x, m) = train.images.T.shape
    n_y = train.labels.T.shape[0]
    
    costs = []
    
    # Create placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(hidden_1, hidden_2, n_input, n_classes)
    
    # Forward propagation
    Z3 = forward_propagation(X, parameters)

    # Cost function
    cost = compute_cost(Z3, Y, labels)

    # Backpropagation (using Adam optimizer)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    
    # Start session to compute Tensorflow graph
    with tf.Session() as sess:
        
        # initialization
        sess.run(init)
        
        # training loop
        for epoch in range(num_epochs):
            
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            
            for i in range(num_minibatches):
                
                # next batch of training data and labels
                minibatch_X, minibatch_Y = train.next_batch(minibatch_size)
                
                # optimizer and cost function
                _, minibatch_cost = sess.run(
                    [optimizer, cost],
                    feed_dict={X: minibatch_X.T, Y: minibatch_Y.T}
                )
                
                # update cost
                epoch_cost += minibatch_cost / num_minibatches
                
            # cost every epoch
            if print_cost == True:
                print('Cost after epoch {epoch_num}: {cost}'.format(
                    epoch_num=epoch,
                    cost=epoch_cost
                ))
                costs.append(epoch_cost)
        
        # plot costs
        plt.figure(figsize=(16,5))
        plt.plot(np.squeeze(costs), color='#2A688B')
        plt.xlim(0, num_epochs-1)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title('learning rate = {rate}'.format(rate=learning_rate))
        plt.show()
        
        # save parameters
        parameters = sess.run(parameters)
        print('Parameters have been trained!')
        
        # calculate correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # calculate accuracy on test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
        print ('Train Accuracy:', accuracy.eval({
            X: train.images.T,
            Y: train.labels.T
        }))
        print ('Test Accuracy:', accuracy.eval({
            X: test.images.T,
            Y: test.labels.T
        }))
