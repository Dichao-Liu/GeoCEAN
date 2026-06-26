"""
Usage (example):
  python train.py --epochs 300 --seed 0 --fold 0
"""
from __future__ import print_function
import os
from my_utils import *
import cv2
import torch
import torchvision
import torch.optim as optim
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from whitebox_cervical_model import WhiteBoxEnergyNet  
import argparse

parser = argparse.ArgumentParser(description='Organize Dataset')
parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--d', type=str, default='d', choices=['d','h','s'], help='dataset: d=DSCC, h=Herlev, s=SIP')
args, unparsed = parser.parse_known_args()

seed = args.seed
input_size = 224
seed_everything(seed)


def inference(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    net.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    score_list, target_list, pred_list = [], [], []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)

        score_list.append(outputs.softmax(dim=1).data.cpu())
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.data.cpu().unsqueeze(0))
        target_list.append(targets.data.cpu().unsqueeze(0))

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0 or batch_idx == (len(test_loader)-1):
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    pred_list = torch.cat(pred_list, axis=-1).squeeze().numpy()
    target_list = torch.cat(target_list, axis=-1).squeeze().numpy()
    score_list = torch.cat(score_list, axis=0).squeeze().numpy()

    accuracy = accuracy_score(pred_list, target_list)*100
    f1_micro = f1_score(target_list,pred_list,average='micro')
    f1_macro = f1_score(target_list,pred_list,average='macro')
    auc_micro = roc_auc_score(target_list, score_list, multi_class='ovr',average='micro')
    auc_macro = roc_auc_score(target_list, score_list, multi_class='ovr',average='macro')

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)
    print("Test Accuracy: {}%".format(test_acc))

    return accuracy, f1_micro, f1_macro, auc_micro, auc_macro


def test(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    test_acc = 100. * float(correct) / total
    test_loss = test_loss / (idx + 1)
    return test_acc, test_loss


def train(nb_epoch, batch_size, num_class, store_name, lr=0.002, data_path='', start_epoch=0, test_folder=''):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print("use_cuda:", use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path,'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # >>> minimal change: use WhiteBoxStarWave (same signature)
    net = WhiteBoxEnergyNet(num_class)

    netp = torch.nn.DataParallel(net)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0

    for epoch in tqdm(range(start_epoch, nb_epoch)):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            inputs, targets = inputs.to(device), targets.to(device)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr)

            optimizer.zero_grad()
            outputs = netp(inputs)
            loss = CELoss(outputs, targets)
            loss.backward()
            optimizer.step()

            #  training log
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            train_loss += loss.item()

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write('Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (epoch, train_acc, train_loss))

        val_acc, val_loss = test(net, CELoss, 4, os.path.join(data_path, test_folder))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (epoch, val_acc, val_loss))

    trained_model = torch.load('./' + store_name + '/model.pth')
    accuracy, f1_micro, f1_macro, auc_micro, auc_macro = inference(trained_model, CELoss, 4, os.path.join(data_path, 'validation'))
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f \\n' % (
        accuracy, f1_micro, f1_macro, auc_micro, auc_macro))



if __name__ == '__main__':
    seed_everything(seed)
    lr = 0.004

    # Results dirs
    results_path = 'results'
    mk_dir(results_path)
    task_result_path = os.path.join(results_path, 'classification')
    mk_dir(task_result_path)

    # Dataset routing
    if args.d == 'd':
        # DSCC
        data_path = f"datasets/DSCC/splited/seed_0_5fold/{args.fold}"
        num_cls = 3
        dtag = 'DSCC'
    elif args.d == 'h':
        # Herlev
        data_path = f"datasets/Herlev/splited/seed_0_5fold/{args.fold}"
        num_cls = 7
        dtag = 'Herlev'
    else:
        # SIP
        data_path = f"datasets/SIP/splited/seed_0_5fold/{args.fold}"
        num_cls = 5
        dtag = 'SIP'

    pyname = os.path.basename(__file__).replace('.py','').replace('main_','')
    experiment_result_path = os.path.join(
        task_result_path,
        "{}_{}_seed_{}_fold_{}_input_size_{}_lr_{}_pretrained_{}_epochs_{}".format(
            pyname, dtag, args.seed, args.fold, input_size, lr, int(args.pretrained), args.epochs
        )
    )

    train(nb_epoch=args.epochs,
          batch_size=16,
          num_class=num_cls,
          lr=lr,
          store_name=experiment_result_path,
          data_path=data_path,
          start_epoch=0,
          test_folder='validation')
