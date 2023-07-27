import torch
import os
from utilities import train_test_loader, gen_key_chain
import argparse

KEY_DIM = (28, 32, 32, 224)  # equal to image dimension: MNIST 28， CIFAR10 32， CIFAR100 32， FOOD101 224

def compare_model(model, ref_model, dataloader, device):
    import copy
    import torch.nn.functional as F
    model.to(device)
    ref_model.to(device)
    model.eval()
    ref_model.eval()

    weight_loss = 0.0
    for (n1, p1), (_, p2) in zip(ref_model.named_parameters(), model.named_parameters()):
        print('Name:{}, Diff: {:.4f}.'.format(n1, (p1 - p2).norm()))
        weight_loss += (p1 - p2).norm(p=2).item()**2

    features = torch.empty(0, model.out.weight.shape[1]+1).to(device)
    features_ref = torch.empty(0, model.out.weight.shape[1]+1).to(device)

    prototype_ref = copy.copy(ref_model.out.weight.detach()).to('cpu')
    prototype = copy.copy(model.out.weight.detach()).to('cpu')

    prototype_loss = (prototype - prototype_ref).norm(p=2) ** 2

    correct = 0
    correct_ref = 0
    feature_loss = 0.0
    kld_loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            if data.shape[1] == 1:
                data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            feature, pred = model(data)
            feature_ref, pred_ref = ref_model(data)

            feature_loss += F.mse_loss(feature, feature_ref, reduction='sum')
            kld_loss += F.kl_div(F.log_softmax(pred, dim=1), F.softmax(pred_ref, dim=1), reduction='sum')

            correct += pred.argmax(dim=1).eq(target).sum().item()
            correct_ref += pred_ref.argmax(dim=1).eq(target).sum().item()

            t1 = torch.cat((feature, target.unsqueeze(-1)), dim=1)
            features = torch.cat((features, t1), dim=0)

            t2 = torch.cat((feature_ref, target.unsqueeze(-1)), dim=1)
            features_ref = torch.cat((features_ref, t2), dim=0)

    feature_loss = feature_loss / features.shape[0]
    kld_loss = kld_loss / features.shape[0]

    print('{}\n{}\n{}\n{}'.format(weight_loss, feature_loss.item(), kld_loss.item(), prototype_loss.item()))
    return weight_loss, feature_loss.item(), kld_loss.item(), prototype_loss.item()


if __name__ == '__main__':

    """
    embed_mode (0-4 studied in [1], 5-9 studied in [2]):
    0: ref          no backdoor embedding, to train a reference model
    1: scratch      embed from scratch
    2: FTAL         embed via fine-tune all layers (FTAL) 
    3: FTLL         embed via fine-tune last layer (FTLL)
    4: FTAL+PGR     embed via FTAL plus the proposed prototype guided regularizer (PGR)
    5: FTAL+TWL     embed via FTAL plus total weight loss (TWL) (the TWL mimics the embedding distortion in multimedia watermarking)
    6: FixLL        embed via fix last layer (FixLL) (fix the classifier of the reference model)
    7: FixLL+TWL    embed via FixLL plus TWL
    8: FixLL+PFL    embed via FixLL plus the proposed penultimate feature loss (PFL) (achieving deep fidelity)
    9: FixLL+SPL    embed via FixLL plus the proposed softmax probability-distribution loss (SPL) (achieving deep fidelity)

    [1] G. Hua, A. B. J. Teoh, Y. Xiang, and H. Jiang, "Unambiguous and high-fidelity backdoor watermarking for deep neural networks," IEEE Transactions on Neural Networks and Learning Systems, 2023.
    
    [2] G. Hua and A. B. J. Teoh, "Deep fidelity in DNN watermarking: A study of backdoor watermarking for classification models," Pattern Recognition, 2023.

    """

    parser = argparse.ArgumentParser(description='DNN Backdoor Watermarking Parameters')
    parser.add_argument('--dataset_index', default=0, type=int, help='0: MNIST, 1: CIFAR10, 2: CIFAR100, 3: FOOD101')
    parser.add_argument('--embed_mode', default=0, type=int, help='0: ref, 1: scratch, 2: FTAL, 3: FTLL, 4: FTAL+PGR, 5: FTAL+TWL, 6: FixLL, 7: FixLL+TWL, 8: FixLL+PFL, 9: FixLL+SPL')
    parser.add_argument('--n', default=10, type=int, help='Number of original trigger samples')
    parser.add_argument('--m', default=10, type=int, help='Length of n trigger chains')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch_size of normal samples')
    parser.add_argument('--mix', default=4, type=int, help='Number of mixed trigger samples per batch')
    parser.add_argument('--dataset_path', default='./data/', type=str, help='Path to store data')
    parser.add_argument('--trained_path', default='./trained/', type=str, help='Path to store trained models')
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    if not os.path.exists(args.trained_path):
        os.makedirs(args.trained_path)

    # TODO: Select dataset and prepare data and trigger
    dataset_name = {0: 'MNIST', 1: 'CIFAR10', 2: 'CIFAR100', 3: 'FOOD101'}  # corresponding model: [ResNet18_p2, ResNet18_p64, wrn_28_10, efficientnet]
    mode = {0: 'ref', 1: 'scratch', 2: 'FTAL', 3: 'FTLL', 4: 'FTAL+PGR', 5: 'FTAL+TWL', 6: 'FixLL', 7: 'FixLL+TWL', 8: 'FixLL+PFL', 9: 'FixLL+SPL'}

    trainloader, testloader = train_test_loader(dataset_name[args.dataset_index], args.dataset_path, batch_size=args.batch_size)

    # TODO: Generate noise pattern key chain trigger set
    if not os.path.exists('./key_chain/trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt'):
        gen_key_chain(dim=KEY_DIM[args.dataset_index], n=args.n, m=args.m, save=True)
    else:
        print(f'trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt already exists.')
    trigger_set = torch.load('./key_chain/trigger_key_chain_' + str(KEY_DIM[args.dataset_index]) + '_' + str(args.n) + '_' + str(args.m) + '.pt')
    trigger_sample, trigger_label = trigger_set['data'], trigger_set['target']

    # TODO: Train and embed
    import train_backdoor
    train_backdoor.train(trainloader, testloader, trigger_sample, trigger_label, args.trained_path,
                         dataset=dataset_name[args.dataset_index], mode=mode[args.embed_mode], n=args.n, m=args.m, mix=args.mix)


    # # TODO: Compare model and visualization
    # import model_resnet
    # test_model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
    # ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
    # test_path = '/home/ghua/Desktop/dnn_watermark/trained/INSTANCE/CIFAR10/FTAL/CIFAR10_ResNet18_FTAL_Epoch_40_test_acc_85.99%_trigger_acc_100.00%_n_10_m_10_mix_(32)4_LP_0.7091.pt'
    # ref_path = './ref_models/deep_fidelity/CIFAR10_ResNet18_Host_Model_Epoch_37_test_acc_86.44%_trigger_acc_11.00%.pt'
    
    # test_model.load_state_dict(torch.load(test_path), strict=True)
    # ref_model.load_state_dict(torch.load(ref_path), strict=True)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # compare_model(test_model, ref_model, trainloader, device)

    print('End of Program')
