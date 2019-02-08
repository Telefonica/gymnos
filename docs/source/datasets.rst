###############################
Datasets
###############################

.. _datasets:

This section shows a collection of datasets currently supoorted by Gymnos

***********************
Kaggle
***********************
Gymnos currently supports different datasets from Kaggle

Requirements
---------------------
Kaggle uses a private cli to allow users to download published datasets.
A ``username`` and ``key`` are required to interact with Kaggle datasets via Gymnos.
Please visit `kaggle`_  and complete the registration procedure.
Then provide Gymnos with the obtained ``username`` and ``key`` in two ways:

.. _kaggle: https://www.kaggle.com/

Via experiment configuration:

.. code-block:: json

   {
      "dataset": {
         "properties": {
            "service": {
               "credentials": {
                  "username": "example_user_id",
                  "key": "XXXXXXXXXXYYYYYYYYYY"
               }
            }
         }
      }
   }
      
Via ``kaggle.json`` file:

.. code-block:: json

   { "username": "example_user_id", "key": "XXXXXXXXXXYYYYYYYYYY" }

.. note::
   Make sure it's located in **/root/.kaggle**:
  

Dogs vs Cats
---------------------
The training archive contains 25,000 images of dogs and cats. 
Train your algorithm on these files and predict the labels for test1.zip (1 = dog, 0 = cat).
Please visit Kaggle website `dogs-vs-cats`_ for more details.

.. _dogs-vs-cats: https://www.kaggle.com/c/dogs-vs-cats


Config example:

.. code-block:: json

   {
      "dataset": {
         "id": "kaggle-dogs-vs-cats",
         "properties": {
            "service": {
               "credentials": {
                  "username": "example_user_id",
                  "key": "XXXXXXXXXXYYYYYYYYYY"
               },
               "type": "competitions",
               "id": "dogs-vs-cats"
            },
            "type": "images",
            "image_width": 150,
            "image_height": 150,
            "image_depth": 3,
            "color_mode": "multi",
            "class_mode": "binary",
            "batch_size": 512
         }
      }
   }