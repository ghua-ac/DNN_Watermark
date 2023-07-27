import torch
from utilities import bf_attack

""" Notes:
 1. Assign the path of a trained model to 'model_path'
 2. If calling wrn28_10, manually change last layer name to 'out', disable bias, and adjust output into (feature, probability)
"""

if __name__ == '__main__':
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
    model_path = []
    model.load_state_dict(torch.load(model_path))
    
    results = bf_attack(model, 3, 1, 224)

    from homura.vision.models.cifar_resnet import wrn28_10  # manually change last layer name to 'out', disable bias, adjust output
    model = wrn28_10(num_classes=100)
    model_path = []
    model.load_state_dict(torch.load(model_path))

    result1 = bf_attack(model, 1, 3, 32)
    result2 = bf_attack(model, 2, 2, 32)
    result3 = bf_attack(model, 3, 1, 32)
    result4 = bf_attack(model, 5, 1, 32)

print('End of Program.')