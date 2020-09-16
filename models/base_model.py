from abc import ABC, abstractmethod
import tensorflow as tf

class ModelBase(ABC):
    def __init__(self, gpu_initialized=False, training=False, limit=256):
        self.training = training 
        # ensure that tensorflow is executing eagerly. since tensorflow 2 this should be enabled by default.
        assert tf.executing_eagerly()

        # fix for memory growth. is sometimes not initialized.
        if gpu_initialized == False:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if training == True:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            else:
                # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        physical_devices[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
        
    @abstractmethod
    def compile(self, optimizer, loss, metrics, show_summary):
        pass
    
    @abstractmethod
    def fit(self, training, callbacks, epochs, validation, validation_steps, steps_per_epoch):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, test):
        return self.model.evaluate(test)
    
    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def get_layers(self):
        return self.model.layers

    def get_metric_names(self):
        return self.model.metrics_names