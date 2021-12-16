import torch
import os
from utilities import train_test_loader, gen_key_chain
import argparse

if __name__ == '__main__':

    KEY_DIM = (28, 32, 32, 224)
    parser = argparse.ArgumentParser(description='DNN Backdoor Watermarking Parameters')
    parser.add_argument('--dataset_index', default=3, type=int, help='0: MNIST, 1: CIFAR10, 2: CIFAR100, 3: FOOD101')
    parser.add_argument('--mode_index', default=1, type=int, help='0: ref, 1: bd_scratch, 2: bd_FTAL, 3: bd_FTLL, 4: bd_ref_guide')
    parser.add_argument('--n', default=10, type=int, help='Number of original trigger samples')
    parser.add_argument('--m', default=100, type=int, help='Length of n trigger chains')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch_size of normal samples')
    parser.add_argument('--mix', default=4, type=int, help='Number of mixed trigger samples per batch')
    parser.add_argument('--dataset_path', default='./data/', type=str, help='Path to store data')
    parser.add_argument('--trained_path', default='./trained/', type=str, help='Path to store trained models')
    args = parser.parse_args()
    assert args.dataset_index in range(4), \
        f"Invalid dataset."
    assert args.mode_index in range(5), \
        f"Invalid mode."

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.trained_path):
        os.makedirs(args.trained_path)

    # TODO: Select dataset and prepare data and trigger
    dataset_name = {0: 'MNIST', 1: 'CIFAR10', 2: 'CIFAR100', 3: 'FOOD101'}  # corresponding model: [ResNet18_p2, ResNet18_p64, wrn_28_10, efficientnet]
    mode = {0: 'ref', 1: 'bd_scratch', 2: 'bd_FTAL', 3: 'bd_FTLL', 4: 'bd_ref_guide'}
    trainloader, testloader = train_test_loader(dataset_name[args.dataset_index], args.dataset_path, batch_size=args.batch_size)

    if not os.path.exists('./key_chain/trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt'):
        gen_key_chain(dim=KEY_DIM[args.dataset_index], n=args.n, m=args.m, save=True)
    else:
        print(f'trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt already exists.')
    trigger_set = torch.load('./key_chain/trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt')
    trigger_sample, trigger_label = trigger_set['data'], trigger_set['target']

    #  TODO: Train and embed
    import train_backdoor
    train_backdoor.train(trainloader, testloader, trigger_sample, trigger_label, args.trained_path,
                         dataset=dataset_name[args.dataset_index], mode=mode[args.mode_index], n=args.n, m=args.m, mix=args.mix)

    # TODO: Compare model
    # from efficientnet_pytorch import EfficientNet
    # test_model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
    # ref_model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
    #
    # # test_path = trained_path + 'FOOD101_EfficientNet-B0_bd_scratch_Epoch_24_test_acc_81.24%_trigger_acc_100.00%.pt'
    # test_path = trained_path + 'FOOD101_EfficientNet-B0_bd_FTAL_Epoch_23_test_acc_81.31%_trigger_acc_100.00%_mix_8.pt'
    # ref_path = 'ref_models/FOOD101_EfficientNet-B0_ref_Epoch_87_test_acc_80.65%_trigger_acc_0.00%.pt'
    #
    # test_model.load_state_dict(torch.load(test_path))
    # ref_model.load_state_dict(torch.load(ref_path))
    #
    # print('lp: {:.4f},'.format((ref_model.out.weight.detach() - test_model.out.weight.detach()).norm()))

    print('End of Program')
