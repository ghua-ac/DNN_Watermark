import torch
from utilities import bf_attack
from efficientnet_pytorch import EfficientNet


if __name__ == '__main__':
    model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
    model_path = './trained/FOOD101_EfficientNet-B0_bd_ref_guide_Epoch_48_test_acc_73.66%_trigger_acc_100.00%_n_10_m_100_mix_(32)8_LP_0.0035.pt'
    model.load_state_dict(torch.load(model_path))

    results = bf_attack(model, 3, 1, 224)

    print('End of Program.')
