#
#
#   Test
#
#

import matplotlib.pyplot as plt

from PIL import Image

from gymnos.vision.object_detection.yolov4 import Yolov4Predictor

predictor: Yolov4Predictor = Yolov4Predictor.from_pretrained("ruben/models/chinese-coins-detector")
img = Image.open("/Users/rubens/.gymnos/datasets/coins_detection/P00524-152122.jpg")

predictions = predictor.predict(img)

print(predictions)

new_img = predictor.plot(img, predictions)
plt.imshow(new_img)
plt.show()
