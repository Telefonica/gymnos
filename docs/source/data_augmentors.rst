###################
Data Augmentors
###################

Gymnos data augmentors are a collection of data augmentors with a common API allowing their use in a pipeline of a supervised learning system. All data augmentors inherit from :class:`gymnos.data_augmentors.data_augmentor.DataAugmentor`.

Usage
*******
.. code-block:: python

    data_augmentor = gymnos.data_augmentors.load("flip", probability=0.2, top_bottom_left_right="LEFT_RIGHT")

    new_image = data_augmentor.transform(image)  # transform image

If you want to use multiple data augmentors or transform according to a probability, take a look to Pipeline:

.. code-block:: python

    data_augmentor_1 = gymnos.data_augmentors.load("histogram_equalisation", probability=0.3)
    data_augmentor_2 = gymnos.data_augmentors.load("invert", probability=0.4)

    pipeline = gymnos.data_augmentors.data_augmentor.Pipeline(
        data_augmentor_1,
        data_augmentor_2
    )

    new_image = pipeline.transform(image)  # transform image with the data augmentors, according to probability

All Data Augmentors
********************

.. contents:: 
    :local: 


distort
==================================
.. autoclass:: gymnos.data_augmentors.distort.Distort
    :noindex:

flip
==================================
.. autoclass:: gymnos.data_augmentors.flip.Flip
    :noindex:

gaussian_distortion
==================================
.. autoclass:: gymnos.data_augmentors.gaussian_distortion.GaussianDistortion
    :noindex:

greyscale
==================================
.. autoclass:: gymnos.data_augmentors.greyscale.Greyscale
    :noindex:

histogram_equalisation
==================================
.. autoclass:: gymnos.data_augmentors.histogram_equalisation.HistogramEqualisation
    :noindex:

invert
==================================
.. autoclass:: gymnos.data_augmentors.invert.Invert
    :noindex:

random_brightness
==================================
.. autoclass:: gymnos.data_augmentors.random_brightness.RandomBrightness
    :noindex:

random_color
==================================
.. autoclass:: gymnos.data_augmentors.random_color.RandomColor
    :noindex:

random_contrast
==================================
.. autoclass:: gymnos.data_augmentors.random_contrast.RandomContrast
    :noindex:

random_erasing
==================================
.. autoclass:: gymnos.data_augmentors.random_erasing.RandomErasing
    :noindex:

rotate
==================================
.. autoclass:: gymnos.data_augmentors.rotate.Rotate
    :noindex:

rotate_range
==================================
.. autoclass:: gymnos.data_augmentors.rotate_range.RotateRange
    :noindex:

shear
==================================
.. autoclass:: gymnos.data_augmentors.shear.Shear
    :noindex:

skew
==================================
.. autoclass:: gymnos.data_augmentors.skew.Skew
    :noindex:

zoom
==================================
.. autoclass:: gymnos.data_augmentors.zoom.Zoom
    :noindex:

zoom_ground_truth
==================================
.. autoclass:: gymnos.data_augmentors.zoom_ground_truth.ZoomGroundTruth
    :noindex:

zoom_random
==================================
.. autoclass:: gymnos.data_augmentors.zoom_random.ZoomRandom
    :noindex:
