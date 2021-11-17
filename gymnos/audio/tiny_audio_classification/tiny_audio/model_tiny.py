
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
from .c_writer import *
import logging


def create_model(self):


    model = Sequential()
    channels = self.channels
    columns = self.columns

    rows = int(1024 / (columns * channels))
    model.add(Reshape((rows, columns, channels), input_shape=(1024, )))
    model.add(Conv2D(8, kernel_size=3, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, kernel_size=3, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(1), padding='same'))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax', name='y_pred'))


    return model



def export_to_lite(model,self):
    #Step needed to export the model to TF lite and MICRO
    logger = logging.getLogger(__name__)
    logger.info("Optimizing for tf lite ...")



    def apply_qat_to_dense_and_cnn(layer):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = tf.keras.models.clone_model(
        model,
            clone_function=apply_qat_to_dense_and_cnn,
            )

    #Optimizing model for TF lite
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    quant_aware_model.summary()

    quant_aware_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=["Accuracy"],
            )

    # Trainning some extra epochs but with optimizations
    logger.info("Trainning with some optimizations...")
    EPOCHS=self.epochs
    quant_aware_history = quant_aware_model.fit(
    self.train_ds,
    validation_data=self.val_ds,
    epochs=EPOCHS
    )

    #Converting to TF lite
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for input_value, output_value in self.val_ds.unbatch().batch(1).take(100):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    with open(self.root+"/"+"tiny_sounds.tflite", "wb") as f:
        f.write(tflite_model_quant)

    c_model_name = "tiny_autoencoder-arduino"


    # Adapting the model to run on C
    hex_array = [format(val, '#04x') for val in tflite_model_quant]
    c_model = create_array(
            np.array(hex_array), 'unsigned char', "micro-model")

    header_str = create_header(c_model, c_model_name)
    logger.info("Model with optimizations ready")
    with open(self.root+"/"+c_model_name + '.h', 'w') as file:
        file.write(header_str)
