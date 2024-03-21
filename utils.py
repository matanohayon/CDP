import numpy as np


def sigmoid(x):
    """
     Parameters
     ----------
     x : np.ndarray input data

     Returns
     -------
     np.ndarray
         sigmoid of the input x
     """
    res=1 / (1 + np.exp(-x))
    return res
    


def sigmoid_prime(x):
    """
         Parameters
         ----------
         x : np.ndarray input data

         Returns
         -------
         np.ndarray
             derivative of sigmoid of the input x

    """
    res=sigmoid(x)*(1-sigmoid(x))
    return res
    


def random_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of xavier initialized np arrays weight matrices
    """
    return [xavier_initialization(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
    


def zeros_weights(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays weight matrices
    """
    return [np.zeros((sizes[i],sizes[i+1])) for i in range(len(sizes)-1)]
    


def zeros_biases(sizes):
    """
         Parameters
         ----------
         sizes : list of sizes

         Returns
         -------
         list of zero np arrays bias matrices
    """

    return [np.zeros(sizes[i])for i in range(len(sizes))]
    


def create_batches(data, labels, batch_size):
    """
         Parameters
         ----------
         data : np.ndarray of input data
         labels : np.ndarray of input labels
         batch_size : int size of batch

         Returns
         -------

         list of tuples of (data batch of batch_size, labels batch of batch_size)
    """
    return \
        [ x for x in
        [(data[batch_size * i : batch_size * (i + 1)], labels[batch_size * i : batch_size * (i + 1)]) for i in range(len(data)//batch_size)] + \
        [(data[batch_size * (len(data)//batch_size) :], labels[batch_size * (len(data)//batch_size) :])] if x[0].shape[0] != 0
        ]




def add_elementwise(list1, list2):
    """
         Parameters
         ----------
         list1 : np.ndarray of numbers
         list2 : np.ndarray of numbers

         Returns
         -------
         list of sum of each two elements by index
    """
    list3=[a+b for a,b in zip(list1,list2)]
    return list3



def xavier_initialization(m, n):
    xavier = 1 / (m ** 0.5)
    return np.random.uniform(low=-xavier, high=xavier, size=(m, n))
