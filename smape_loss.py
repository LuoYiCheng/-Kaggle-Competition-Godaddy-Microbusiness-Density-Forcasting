import tensorflow as tf
import keras.backend as K

# Definte loss
class SMAPE(tf.keras.losses.Loss):
    def __init__(self, name='smape_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        epsilon = 0.1
        summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
        smape = K.abs(y_pred - y_true) / summ * 2.0
        return smape    

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
