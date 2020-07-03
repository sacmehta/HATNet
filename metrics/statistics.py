import sys
import time
from utilities.print_utilities import print_log_message


class Statistics(object):
    '''
    This class is used to store the training and validation statistics
    '''
    def __init__(self):
        super(Statistics, self).__init__()
        self.loss = 0
        self.acc = 0
        self.eps = 1e-9
        self.counter = 1

    def update(self, loss, acc):
        '''
        :param loss: Loss at ith time step
        :param acc: Accuracy at ith time step
        :return:
        '''
        self.loss += loss
        self.acc += acc
        self.counter += 1

    def __str__(self):
        return 'Loss: {}'.format(self.loss)

    def avg_acc(self):
        '''
        :return: Average Accuracy
        '''
        return self.acc / self.counter


    def avg_loss(self):
        '''
        :return: Average loss
        '''
        return self.loss/self.counter

    def output(self, epoch, batch, n_batches, start, lr):
        '''
        Displays the output
        :param epoch: Epoch number
        :param batch: batch number
        :param n_batches: Total number of batches in the dataset
        :param start: Epoch start time
        :param lr: Current LR
        :return:
        '''
        print_log_message(
            "Epoch: {:3d} [{:8d}/{:8d}], "
            "Loss: {:5.2f}, "
            "Acc: {:3.2f}, "
            "LR: {:1.6f}, "
            "Elapsed time: {:5.2f} seconds".format(
                epoch, batch, n_batches,
                self.avg_loss(),
                self.avg_acc(),
                lr,
                time.time() - start
            )
        )
        sys.stdout.flush()