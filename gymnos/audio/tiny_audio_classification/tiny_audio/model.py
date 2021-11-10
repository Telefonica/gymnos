

import tensorflow as tf




def create_model():

    norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
    norm_layer.adapt(cached_ds.map(lambda x, y, z: tf.reshape(x, input_shape)))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.experimental.preprocessing.Resizing(32, 32, interpolation="nearest"),
        norm_layer,
        tf.keras.layers.Conv2D(8, kernel_size=(8,8), strides=(2, 2), activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(6, activation='softmax')
        ])

    return model



def export_to_lite(model):

    annotated_model = tf.keras.models.clone_model(
        model,
            clone_function=apply_qat_to_dense_and_cnn,
            )
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    quant_aware_model.summary()

    quant_aware_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["Accuracy"],
            )

    EPOCHS=1
    quant_aware_history = quant_aware_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for input_value, output_value in train_ds.unbatch().batch(1).take(100):
# Model has only one input so each data point has one element.
            yield [input_value]
    converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model_quant = converter.convert()
    with open("tiny_sounds.tflite", "wb") as f:
        f.write(tflite_model_quant)
