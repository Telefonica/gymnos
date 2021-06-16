#
#
#   Pretrained
#
#

from pprint import pprint
from gymnos.models.vision.image_classification.transfer_efficientnet import TransferEfficientNetPredictor

predictor = TransferEfficientNetPredictor.from_pretrained("fd707f7cf66247008c7c715db36aa31b")

preds = predictor.predict("data/dogs_vs_cats/cat/cat.0.jpg")

print("{:*^30}".format("Prediction"))
print(preds)

print("{:*^30}".format("Classname"))
print(predictor.classes[preds.label])

print("{:*^30}".format("Trainer"))
pprint(predictor.trainer)
