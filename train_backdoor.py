import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
import sys
import model_resnet


def train(trainloader, testloader, trigger_sample, trigger_label, trained_path, dataset='MNIST', mode='ref', n=2, m=10, mix=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {'MNIST': 'ResNet18', 'CIFAR10': 'ResNet18', 'CIFAR100': 'wrn28_10', 'FOOD101': 'EfficientNet-B0'}

    if dataset == 'MNIST':
        model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=True)
        ref_path = 'ref_models/MNIST_ResNet18_ref_Epoch_14_test_acc_99.46%_trigger_acc_10%_ref.pt'

    elif dataset == 'CIFAR10':
        model = model_resnet.resnet18(num_classes=10, penultimate_2d=False)
        ref_model = model_resnet.resnet18(num_classes=10, penultimate_2d=False)
        ref_path = 'ref_models/CIFAR10_ResNet18_ref_Epoch_33_test_acc_86.70%_trigger_acc_9%_ref.pt'

    elif dataset == 'CIFAR100':
        from homura.vision.models.cifar_resnet import wrn28_10  # manually change last layer name to 'out', disable bias, adjust output
        model = wrn28_10(num_classes=100)
        ref_model = wrn28_10(num_classes=100)
        ref_path = 'ref_models/CIFAR100_wrn28_10_ref_Epoch_85_test_acc_75.40%_trigger_acc_0.00%.pt'

    elif dataset == 'FOOD101':
        from efficientnet_pytorch import EfficientNet  # manually change last layer name to 'out', disable bias, adjust output
        model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
        ref_model = EfficientNet.from_name('efficientnet-b0', num_classes=101)
        ref_path = 'ref_models/FOOD101_EfficientNet-B0_ref_Epoch_92_test_acc_74.25%_trigger_acc_0.00%.pt'
    else:
        print('Dataset preparation error.')
        sys.exit()

    num_params = sum(i.numel() for i in model.parameters() if i.requires_grad)
    print('Model number of parameters: {:.0f}'.format(num_params))

    # TODO: Train ref model
    if mode == 'ref':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if dataset == 'MNIST':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
            num_epochs = 20
        elif dataset == 'CIFAR10':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            num_epochs = 50
        elif dataset == 'CIFAR100':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100
        else:  # FOOD101
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100

        for epoch in range(num_epochs):
            model.to(device)
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)
                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                model_str = trained_path + dataset + '_' + model_dict[dataset] + '_' + mode + '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%.pt'\
                    .format(epoch, acc * 100, t_correct / (m * n) * 100)
                torch.save(model.state_dict(), model_str)

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}.'.format(dataset, mode, epoch, acc * 100, t_correct, m * n))
            torch.cuda.empty_cache()

    # TODO: Backdoor embed from scratch
    if mode == 'bd_scratch':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        check_point = torch.load(ref_path)
        ref_model.load_state_dict(check_point, strict=False)
        if dataset == 'MNIST':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
            num_epochs = 20
        elif dataset == 'CIFAR10':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25], gamma=0.1)
            num_epochs = 50
        elif dataset == 'CIFAR100':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
            num_epochs = 100

        for epoch in range(num_epochs):
            model.to(device)
            model.train()
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                model_str = trained_path + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                torch.save(model.state_dict(), model_str)

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()

    # TODO: FTAL and FTLL
    if (mode == 'bd_FTAL') or (mode == 'bd_FTLL'):
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Fine-tuning based, no scheduler
        check_point = torch.load(ref_path)
        model.load_state_dict(check_point, strict=False)
        ref_model.load_state_dict(check_point, strict=False)
        if dataset == 'MNIST':
            num_epochs = 20
        elif dataset == 'CIFAR10':
            num_epochs = 50
        elif dataset == 'CIFAR100':
            num_epochs = 50
        else:
            num_epochs = 50

        for epoch in range(num_epochs):
            model = model.to(device)
            model.train()

            # Freeze batch_norm
            for p in model.modules():
                if isinstance(p, torch.nn.BatchNorm2d):
                    p.eval()
                    p.weight.requires_grad = False
                    p.bias.requires_grad = False

            if mode == 'bd_FTLL':
                # Freeze all except last layer
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                model.out.weight.requires_grad = True

            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                model_str = trained_path + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                torch.save(model.state_dict(), model_str)

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()

    # TODO: FTAL with (CE + prototype guided loss)
    if mode == 'bd_ref_guide':
        print('Dataset: {}, model: {}, mode: {}, training started using {}...'.format(dataset, model_dict[dataset], mode, str(device)))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Fine-tuning based, no scheduler
        check_point = torch.load(ref_path)
        model.load_state_dict(check_point, strict=False)
        ref_model.load_state_dict(check_point, strict=False)

        const_prototype = copy.deepcopy(model.out.weight.detach()).to(device)

        if dataset == 'MNIST':
            num_epochs = 20
        elif dataset == 'CIFAR10':
            num_epochs = 50
        elif dataset == 'CIFAR100':
            num_epochs = 50
        else:
            num_epochs = 50

        for epoch in range(num_epochs):
            model = model.to(device)
            model.train()

            # Freeze batch_norm
            for p in model.modules():
                if isinstance(p, torch.nn.BatchNorm2d):
                    p.eval()
                    p.weight.requires_grad = False
                    p.bias.requires_grad = False

            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if data.shape[1] == 1:
                    data = data.repeat(1, 3, 1, 1)

                trigger_index = torch.randperm(m * n)
                t_data = trigger_sample[trigger_index[0:mix], :, :, :].to(device)
                t_target = trigger_label[trigger_index[0:mix]].to(device)
                data = torch.cat((data, t_data), dim=0)
                target = torch.cat((target, t_target), dim=0)

                optimizer.zero_grad()
                _, output = model(data)

                guided_loss = (model.out.weight - const_prototype).norm(p=2, dim=1).sum()
                alpha = 0.1
                loss = F.cross_entropy(output, target) + alpha * guided_loss
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    if data.shape[1] == 1:
                        data = data.repeat(1, 3, 1, 1)
                    _, output = model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                acc = correct / len(testloader.dataset)

                t_correct = 0
                t_output = torch.zeros(len(trigger_sample), model.out.weight.shape[0], device=device)
                for i in range(len(trigger_sample)):
                    _, t_output[i, :] = model(trigger_sample[i].unsqueeze(0).to(device))
                t_pred = t_output.argmax(dim=1)
                t_correct += t_pred.eq(trigger_label.to(device)).sum().item()

                prototype_ref = copy.copy(ref_model.out.weight.detach())
                prototype = copy.copy(model.to('cpu').out.weight.detach())
                lp = (prototype_ref - prototype).norm()

                model_str = trained_path + dataset + '_' + model_dict[dataset] + '_' + mode + \
                    '_Epoch_{:.0f}_test_acc_{:.2f}%_trigger_acc_{:.2f}%_n_{:.0f}_m_{:.0f}_mix_({:.0f}){:.0f}_LP_{:.4f}.pt' \
                    .format(epoch, acc * 100, t_correct / (m * n) * 100, n, m, trainloader.batch_size, mix, lp)
                torch.save(model.state_dict(), model_str)

                print('{}, Mode: {}, Epoch: {:.0f}, testing acc: {:.2f}%, trigger Acc: {:.0f}/{:.0f}, LP: {:.4f}, mix: ({:.0f}){:.0f}.'
                      .format(dataset, mode, epoch, acc * 100, t_correct, m * n, lp, trainloader.batch_size, mix))
            torch.cuda.empty_cache()
