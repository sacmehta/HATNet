

class DictWriter(object):
    def __init__(self, file_name, format='csv'):
        super(DictWriter, self).__init__()
        assert format in ['csv', 'json', 'txt']

        self.file_name = '{}.{}'.format(file_name, format)
        self.format = format

    def write(self, data_dict: dict):
        if self.format == 'csv':
            import csv
            with open(self.file_name, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in data_dict.items():
                    writer.writerow([key, value])
        elif self.format == 'json':
            import json
            with open(self.file_name, 'w') as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=True)
        else:
            with open(self.file_name, 'w') as txt_file:
                for key, value in data_dict.items():
                    line = '{} : {}\n'.format(key, value)
                    txt_file.write(line)
