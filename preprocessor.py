#
#   @date:  [TODO: 13/03/2024]
#   @author: [TODO: Matanel Ohayon]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
import numpy as np
from scipy import ndimage


class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        print("start worker")
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

        # raise NotImplementedError("To be implemented")

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image

        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape((28, 28))
        image = ndimage.rotate(image, angle, reshape=False)
        image = image.reshape(784)
        return image
        # raise NotImplementedError("To be implemented")

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis

        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape((28, 28))
        image = ndimage.shift(image, (dy, dx))
        image = image.reshape(784)
        return image
        # raise NotImplementedError("To be implemented")

    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        noise_arr = np.random.uniform(-noise, noise, (image.shape))
        image = image + noise_arr
        image[image > 1] = 1.0
        image[image < 0] = 0.0

        return image

        raise NotImplementedError("To be implemented")

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        image = image.reshape((28, 28))
        result = np.zeros_like(image)
        for i in range(28):
            for j in range(28):
                if 0 <= (j+i*tilt) < 28:
                    result[i][j] = image[i][int(j+i*tilt)]
                else:
                    result[i][j] = 0

        result = result.reshape(784)
        return result

        # raise NotImplementedError("To be implemented")

    def process_image(self, image):
        '''Apply the image process functions
                Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        angle = np.random.randint(-15, 10)
        dx = np.random.randint(0, 2)
        dy = np.random.randint(0, 2)
        noise = np.random.uniform(0.0, 0.15)
        tilt = np.random.uniform(-0.2, 0.2)

        #image = self.shift(image, dx, dy)
        image = self.add_noise(image, noise)
        image = self.shift(image, dx, dy)
        image = self.rotate(image, angle)
        image = self.skew(image, tilt)
        #image = self.add_noise(image, noise)

        return image

        # raise NotImplementedError("To be implemented")

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
                Hint: you can either generate (i.e sample randomly from the training data)
                the image batches here OR in ip_network.create_batches
        '''

        print("strat_running")
        while True:
            next_job = self.jobs.get()
            if next_job is None:  # Poison pill means shutdown
                self.jobs.task_done()
                break
            image = next_job[0]
            processed_image = self.process_image(image)

            self.jobs.task_done()
            self.result.put((processed_image, next_job[1]))

        # raise NotImplementedError("To be implemented")
