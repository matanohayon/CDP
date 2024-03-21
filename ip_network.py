#
#   @date:  [TODO: 13/03/2024]
#   @author: [TODO: Matanel Ohayon]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import os
import sys
from multiprocessing import Queue, JoinableQueue
from network import *
from preprocessor import *
from my_queue import *


class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        print("start fit", training_data[0].size)
        self.jobs = JoinableQueue()
        #self.result = Queue()
        self.result = MyQueue()

        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)

        self.num_workers = multiprocessing.cpu_count()
        # self.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        workers = [Worker(self.jobs, self.result, training_data,
                          self.mini_batch_size) for i in range(self.num_workers)]

        for worker in workers:
            worker.start()

        # 2. Set jobs
        self.num_tasks = len(training_data[0])
        print("num_task ", self.num_tasks)
        for i in range(self.num_tasks):
            data = (training_data[0][i], training_data[1][i])
            self.jobs.put(data)

        for _ in range(self.num_workers):
            self.jobs.put(None)

        image_data = list(training_data[0])
        label_data = list(training_data[1].copy())
        for i in range(self.num_tasks):
            image = self.result.get()
            # print("image ", image[0].size)
            image_data.append(image[0])
            label_data.append(image[1])

        #print("image_data ", len(image_data))

        image_data = np.array(image_data)
        label_data = np.array(label_data)
        training_data = ((image_data, label_data))

        s = len(training_data[0])

       # for i in range(s):
        #    print(training_data[0][i].size)

        print("Got here ", s)

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        self.jobs.join()

        # 3. Stop Workers
        for worker in workers:
            worker.join()

        # raise NotImplementedError("To be implemented")

    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
                Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''

        print("start batches")

        batches = []
        for k in range(self.number_of_batches):
            indexes = random.sample(range(0, data.shape[0]), batch_size)
            images = data[indexes]
            lbls = labels[indexes]
            batches.append((images, lbls))

        return batches
        # raise NotImplementedError("To be implemented")
