#
#
#   Detect chinese coins
#
#

from gymnos.utils import assert_dependencies

assert_dependencies([
    "matplotlib",
    "gymnos[vision.object_detection.yolov4]"
])

import click
import matplotlib.pyplot as plt

from gymnos.vision.object_detection.yolov4 import Yolov4Predictor


@click.command()
@click.argument("img_path")
def predict(img_path):
    predictor = Yolov4Predictor.from_pretrained("ruben/models/chinese-coins-detector")
    predictions = predictor.predict(img_path)

    new_img = predictor.plot(img_path, predictions)

    plt.imshow(new_img)
    plt.show()


if __name__ == "__main__":
    predict()
