#
#
#   Keras Network Builder
#
#

from keras import models, layers


def create_layer(layer_specs):
    layer_type = layer_specs["type"]
    if layer_type == "dense":
        return layers.Dense(units=layer_specs["size"], activation=layer_specs.get("activation"),
                            kernel_initializer=layer_specs.get("kernel_initializer", "glorot_uniform"))
    elif layer_type == "conv2d":
        return layers.Conv2D(filters=layer_specs["size"], kernel_size=layer_specs["window"],
                             activation=layer_specs.get("activation"), strides=layer_specs.get("strides", 1),
                             padding=layer_specs.get("padding", "valid"))
    elif layer_type == "flatten":
        return layers.Flatten()
    elif layer_type == "dropout":
        return layers.Dropout(rate=layer_specs["rate"])
    elif layer_type == "maxpooling2d":
        return layers.MaxPooling2D(pool_size=layer_specs["size"])


def create_network(network_specs, input_shape=None):
    input_layer = layers.Input(shape=input_shape)

    output_layer = input_layer
    for layer_spec in network_specs:
        output_layer = create_layer(layer_spec)(output_layer)

    return models.Model(inputs=[input_layer], outputs=[output_layer])
