#
#
#   Classify dog or cat
#
#

from gymnos.utils import assert_dependencies

assert_dependencies([
    "gymnos[vision.image_classification.transfer_efficientnet]"
])

import click
import inspect

from gymnos.vision.image_classification.transfer_efficientnet import TransferEfficientNetPredictor


@click.command()
@click.argument("img_path")
def predict(img_path):
    predictor = TransferEfficientNetPredictor.from_pretrained("ruben/models/dogs-vs-cats-4")
    pred = predictor.predict(img_path)

    print(inspect.cleandoc(f"""
        Class: {predictor.classes[pred.label]}
        Label: {pred.label}
        Score: {pred.probabilities[pred.label]}
    """))


if __name__ == "__main__":
    predict()
