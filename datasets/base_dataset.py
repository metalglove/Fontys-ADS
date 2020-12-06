import tensorflow as tf
from abc import ABC, abstractmethod

class DatasetBase(ABC):
    def split_data_to_train_val_test(self, data, train_percentage, validation_percentage, test_percentage, data_size=-1):
        # size of the data
        if data_size == -1:
            data_size = tf.data.experimental.cardinality(data).numpy()

        # prepares train, validation, and test sizes
        self.train_size = int(train_percentage * data_size)
        self.val_size = int(validation_percentage * data_size)
        self.test_size = int(test_percentage * data_size)

        percentage = round(train_percentage + validation_percentage + test_percentage, 8)
        print('train:', self.train_size, 'validation:', self.val_size, 'test:', self.test_size)
        assert percentage == 1.0, f'train, validation, and test size do not sum up to data size: {percentage}'

        # creates the train, validation, and test datasets
        self.train_ds = data.take(self.train_size)
        self.test_ds = data.skip(self.train_size)
        self.val_ds = self.test_ds.take(self.val_size)
        self.test_ds = self.test_ds.skip(self.test_size)

        # ensures that the datasets are cached and repeatable
        self.train_ds = self.train_ds.cache().repeat()
        self.val_ds = self.val_ds.cache().repeat()
        self.actual_test_ds = self.test_ds.take(self.test_size)
        self.test_ds = self.test_ds.skip(self.test_size).cache().repeat()#.as_numpy_iterator()
    
    def shuffle(self, buffer_size):
        # shuffles the dataset and batches it to the batch_size
        self.data = self.data.shuffle(buffer_size)
        if self.batch_size > 0:
            self.data = self.data.batch(self.batch_size)

