"""
Image classifier based on Transfer Learning with EfficientNet as backbone
"""

from ....utils import lazy_import

# Public API
TransferEfficientNetPredictor = lazy_import("gymnos.vision.image_classification.transfer_efficientnet.predictor."
                                            "TransferEfficientNetPredictor")
