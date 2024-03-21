#
#   @date:  [ 14/3/2024]
#   @author:  [Matanel Ohayon]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from multiprocessing import Lock, Pipe, Value


class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.lock = Lock()
        self.parent, self.child = Pipe()
        self.value = Value('i', 0)

        # raise NotImplementedError("To be implemented")

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.lock.acquire()
        self.child.send(msg)
        self.value.value += 1
        self.lock.release()

        # raise NotImplementedError("To be implemented")

    def get(self):
        '''Get the next message from queue (FIFO)

        Return
        ------
        An object
        '''
        msg = self.parent.recv()
        self.value.value -= 1
        return msg
        # raise NotImplementedError("To be implemented")

    def empty(self):
        '''Get whether the queue is currently empty

        Return
        ------
        A boolean value
        '''

        if self.value.valuec == 0:
            return True
        return False

        # raise NotImplementedError("To be implemented")
