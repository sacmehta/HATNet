# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"


# ============================================

class ColorEncoder(object):
    def __init__(self):
        super(ColorEncoder, self).__init__()

    def get_colors(self, dataset_name):
        if dataset_name == 'bbwsi':
            class_colors = [
                (228/ 255.0, 26/ 255.0, 28/ 255.0),
                (55/ 255.0, 126/ 255.0, 184/ 255.0),
                (77/ 255.0, 175/ 255.0, 74/ 255.0),
                (152/ 255.0, 78/ 255.0, 163/ 255.0)
            ]

            class_linestyle = ['solid', 'solid', 'solid', 'solid']

            return class_colors, class_linestyle
        else:
            raise NotImplementedError
