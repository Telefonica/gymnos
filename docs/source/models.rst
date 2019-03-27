###############################
Models
###############################

.. _models:

This section shows a collection of models currently supported by Gymnos

***********************
VGG16
***********************
A wrapper for keras implementation of VGG16 model

Basic Config
---------------------
Add this configuration to your experiment json file

.. code-block:: json

   {
        "model": {
            "name": "VGG16",
            "id": "vgg16",
            "options": {
                "compilation": {
                    "optimizer": "rmsprop",
                    "loss": "categorical_crossentropy",
                    "metrics": ["accuracy"]
                }
            }
        }
    }

Fine Tunning
---------------------
Our implementation of VGG16 supports fine-tunning by configuration.

.. code-block:: json

   {
        "model": {
        "name": "VGG16",
        "id": "vgg16",
        "options": {
            "compilation": {
                "optimizer": "rmsprop",
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"]
            },
            "custom":{
                "fine-tunning": {
                    "input_width": 48,
                    "input_height": 48,
                    "input_depth": 3,
                    "batch_size": 512,
                    "extra_layers":{
                        "classifier": { 
                            "relu": { 
                                "alpha": 0.1 
                            }
                        }
                    }
                }
            }
        }
    }

.. note::

    VGG16 model is pretrained with ImageNet dataset weights by default

.. warning::

    Make sure you specify suitable extra layers to add on top of the pretrained network


***********************
Custom Stack
***********************
Allows to implement via configuration a model based on a stack of layers  

Basic Config
---------------------
Add this configuration to your experiment json file

.. code-block:: json

    {
        "model": {
            "name": "CustomStack",
            "id": "custom-stack",
            "options": {
                "compilation": {
                    "optimizer": "rmsprop",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"]
                },
                "custom":{
                    "framework": "keras",
                    "stack": {
                        "layers": [
                            { "type":"convolutional2D", "settings": { "filter": 32, "kernel_size":[3, 3], "activation": "relu", "input_shape": [ 150, 150, 3 ] } },
                            { "type":"maxpooling2D", "settings": { "pool_size":[2, 2] } },
                            { "type":"convolutional2D", "settings": { "filter": 64, "kernel_size":[3, 3], "activation": "relu" } },
                            { "type":"maxpooling2D", "settings": { "pool_size":[2, 2] } },
                            { "type":"convolutional2D", "settings": { "filter": 128, "kernel_size":[3, 3], "activation": "relu" } },
                            { "type":"maxpooling2D", "settings": { "pool_size":[2, 2] } },
                            { "type":"convolutional2D", "settings": { "filter": 128, "kernel_size":[3, 3], "activation": "relu" } },
                            { "type":"maxpooling2D", "settings": { "pool_size":[2, 2] } },
                            { "type":"flatten", "settings": { }},
                            { "type":"dense", "settings": { "units": 512, "activation": "relu" } },
                            { "type":"dense", "settings": { "units": 1, "activation": "sigmoid" } }
                        ]
                    }
                }
            }
        }
    }
