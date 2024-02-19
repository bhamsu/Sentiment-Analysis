""" Main class, or root class to call other functions. """

# Importing all the required global as well local packages
try:
    import pandas as pd
    import numpy as np
    from src.process_text import process_text_data
except DeprecationWarning:
    print('Deprecation Warning...')

class main:
    def __init__(self, dataset_path):
        column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
        raw_data = pd.read_csv(dataset_path, names = column_names, encoding = 'ISO-8859-1')
        # print(raw_data.head())
        self.data = process_text_data(pd.DataFrame(raw_data[['text', 'target']]))()
        # print(self.data.head())

if __name__ == "__main__":
    main("data/training.1600000.processed.noemoticon.csv")
