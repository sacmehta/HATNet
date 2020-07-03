from utilities.save_dict_to_file import DictWriter
import os
from utilities.print_utilities import *


class SummaryWriter(object):
    def __init__(self, log_dir, format='csv', *args, **kwargs):
        super(SummaryWriter, self).__init__()
        self.summary_dict = dict()
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.file_name = '{}/logs'.format(log_dir)
        self.dict_writer = DictWriter(file_name=self.file_name, format=format)
        self.step = 20

    def add_scalar(self, tag, value, step=None, *args, **kwargs):
        if tag not in self.summary_dict:
            self.summary_dict[tag] = [(value, step)]
        else:
            self.summary_dict[tag].append((value, step))

    def close(self, *args, **kwargs):
        self.dict_writer.write(self.summary_dict)
        try:
            from matplotlib import pyplot as plt
            for k, v in self.summary_dict.items():
                y_axis = []
                x_axis = []
                for val, step in v:
                    y_axis.append(val)
                    if step is not None:
                        x_axis.append(step)
                plt.title(k)
                plt.plot(y_axis)
                if len(x_axis) != 0:
                    assert len(y_axis) == len(x_axis)
                    # filter x_axis to avoid label overlap
                    x_axis = x_axis[0::self.step]
                    plt.xticks(x_axis, rotation=90)
                f_name = '{}/{}.png'.format(self.log_dir, k)
                plt.savefig(f_name, dpi=300, bbox_inches='tight')
                plt.clf()
        except:
            print_warning_message('Matplotlib is not installed so unable to draw plots')


if __name__ == '__main__':
    logger = SummaryWriter('results', comment=None)
    import random

    for i in range(100):
        val = random.random()
        logger.add_scalar('Loss', val, None)
        val = random.random()
        logger.add_scalar('acc', val)

    logger.close()
