# PyTorch
from tkinter import N
from typing import Tuple
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split, Dataset
import torch.nn.functional as F
import torch.nn as nn
# Sklearn
from sklearn.model_selection import train_test_split
# Other
import os
import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy

import argparse
import os


img_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomCrop(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]),
}


def get_dataloades_v2(train_root, test_root, val_root):
    train_data = datasets.ImageFolder(root=train_root)
    test_data = datasets.ImageFolder(root=test_root)
    val_data = datasets.ImageFolder(root=val_root)

    # initial training size
    n_init_train = 3
    # X pool size is the remaining training data
    n_pool = len(train_data) - n_init_train

    # train_data, _, _ = random_split(train_data, [n_train, 0, 0])
    init_train_data, pool_data, _ = random_split(train_data, [n_init_train, n_pool, 0])

    print(len(train_data))
    # print(type(train_data))
    # Convert test data from imagefolder to dataset.Subset
    test_data = torch.utils.data.dataset.Subset(test_data, range(len(test_data)))
    val_data = torch.utils.data.dataset.Subset(val_data, range(len(val_data)))

    print(len(test_data))
    print(len(val_data))

    init_train_data.dataset.transform = img_transforms["train"]
    pool_data.dataset.transform = img_transforms["train"]
    test_data.dataset.transform = img_transforms["test"]
    val_data.dataset.transform = img_transforms["val"]

    return init_train_data, pool_data, test_data, val_data


def get_model(n_classes):
    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')
    multi_gpu = False
    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    return model


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          test_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    train_on_gpu = cuda.is_available()

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    # overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0

        train_acc = 0
        valid_acc = 0
        test_acc = 0

        # Set to training
        model.train()
        # start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            # target.resize_(128, 2)
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
                    # Forward pass
                    output = model(data)
                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)
                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)
                for data, target in test_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
                    # Forward pass
                    output = model(data)
                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    test_loss += loss.item() * data.size(0)
                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    test_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                test_loss = test_loss / len(test_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)
                test_acc = test_acc / len(test_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc, test_loss, test_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                        )
                        # total_time = timer() - overall_start
                        # print(
                        #     f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        # )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc', 'test_loss', 'test_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    # total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    # print(
    #     f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    # )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'test_loss', 'test_acc'])
    return model, history


def train_model(model, train_set, test, val, num_epoch):

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=0.0001)
    model, history = train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        save_file_name='ad-finetuned.pt',
        max_epochs_stop=3,
        n_epochs=num_epoch,
        print_every=1)

    # scores = (history['test_loss'].values, history['test_acc'].values)
    # print('Initial Test Loss: ', scores[0], ' Initial Test Accuracy: ', scores[1])
    return model, history


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
def least_confidence(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           pred_label))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margim_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]


def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    # print(eni[:, 1])
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)


def get_uncertain_samples(y_pred_prob, n_samples, criteria):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\']' % criteria)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def run_ceal(args):
    train_on_gpu = cuda.is_available()

    train_root = args.train_root
    test_root = args.test_root
    val_root = args.val_root
    init_train_data, pool_data, test_data, val_data = get_dataloades_v2(train_root, test_root, val_root)

    print("Finished loading data")

    model = get_model(args.n_classes)

    model, history = train_model(model, init_train_data, test_data, val_data, args.epochs)

    print("Finished initialized model")

    print("Initial length of train data", len(init_train_data))
    print("Initial length of pool_data", len(pool_data))
    acc_lst = []
    stop = False
    for i in range(args.maximum_iterations):
        # Get y_pred_prob for all unlabeled samples
        y_pred_prob = np.empty((0, args.n_classes))
        pool_loader = DataLoader(pool_data, batch_size=64, shuffle=True)
        for x, y in pool_loader:
            if train_on_gpu:
                x, y = x.cuda(), y.cuda()
            pred_prob = model(x)
            # Since the pred_prob is log probabilities, we need to convert to probabilities
            pred_prob = torch.exp(pred_prob)
            # pred_prob is a tensor, convert to numpy array
            pred_prob = pred_prob.detach().cpu().numpy()
            y_pred_prob = np.concatenate((y_pred_prob, pred_prob), axis=0)
        # print("y_pred_prob", y_pred_prob)

        # If the size of init training is 90 stop is True
        if len(init_train_data) >= args.max_train_size:
            stop = True
            break

        # get the uncertain samples
        # fushion lc, ms, en
        _, lc_idx = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size, 'lc')
        _, ms_idx = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size, 'ms')
        _, en_idx = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size, 'en')
        # Set union of the three uncertain samples
        inter_idx = np.union1d(lc_idx, ms_idx)
        un_idx = np.union1d(inter_idx, en_idx)

        # if len(un_idx) > args.max_train_size - len(init_train_data):
        #     un_idx = un_idx[:args.max_train_size - len(init_train_data)]
        #     stop = True
        #
        # # Get the uncertain samples from the pool by subseting the pool since the pool is a dataset
        # unlabeled_subset = torch.utils.data.Subset(pool_data.dataset, un_idx)
        # # print("Length of uncertain samples", len(unlabeled_subset))
        # # Concate to initial training data
        # init_train_data = torch.utils.data.ConcatDataset([init_train_data, unlabeled_subset])
        # # print("Length of initial training data after adding uncertain samples", len(init_train_data))
        # # Remove the uncertain samples from the pool
        # pool_data = torch.utils.data.Subset(pool_data.dataset, np.setdiff1d(np.arange(len(pool_data)), un_idx))
        # # print("Length of Pool data after select uncetain samples", len(pool_data))
        hc_idx = None
        # Remove those index from y_pred_prob
        if args.cost_effective and not stop:
            # get the high confidence samples
            hc_idx, hc_labels = get_high_confidence_samples(y_pred_prob, args.delta)
            # print("Length of Selected high confidence samples", len(hc_idx))
            # Given hc_inx and un_idx, find the intersection
            # intersect = np.intersect1d(hc_idx, un_idx)
            # print("The number of samples that exist in both high confidence and uncertain samples", len(intersect))

            # hc = np.array([[i, l] for i, l in zip(hc_idx, hc_labels) if i not in un_idx])
            # # Only keep one sample from the hc
            # hc = hc[:args.uncertain_samples_size]
            # if hc.size != 0:
            #     # Get high confidence samples from the pool by subseting the pool since the pool is a dataset
            #     if len(hc) > args.max_train_size - len(init_train_data):
            #         hc = hc[:args.max_train_size - len(init_train_data)]
            #         stop = True
            #
            #     hc_subset = torch.utils.data.Subset(pool_data.dataset, hc[:, 0])
            #     # Concate to initial training data
            #     init_train_data = torch.utils.data.ConcatDataset([init_train_data, hc_subset])
            #     # Remove the high confidence samples from the pool
            #     pool_data = torch.utils.data.Subset(pool_data.dataset, np.setdiff1d(np.arange(len(pool_data)), hc[:, 0]))
            #     # print("Length of Pool data after select high confidence samples", len(pool_data))

        # print(type(hc_idx))
        # print(hc_idx)
        # Randomly select
        select_idx = np.random.choice(np.union1d(un_idx, hc_idx), 3, replace=False) if hc_idx is not None else np.random.choice(un_idx, 3, replace=False)
        select_set = torch.utils.data.Subset(pool_data.dataset, select_idx)
        # Concate to initial training data
        init_train_data = torch.utils.data.ConcatDataset([init_train_data, select_set])
        # Remove the selected samples from the pool
        pool_data = torch.utils.data.Subset(pool_data.dataset, np.setdiff1d(np.arange(len(pool_data)), select_idx))


        print("Training data size after AL selection", len(init_train_data))
        print("Pooling data size after AL selection", len(pool_data))
        if i % args.fine_tunning_interval == 0:
            # Train model
            print("######      ****** Fine Tuning ******      ######")
            # print("Length of initial training data after AL selection", len(init_train_data))
            model, history = train_model(model, init_train_data, test_data, val_data, args.epochs)
            args.delta -= (args.threshold_decay * args.fine_tunning_interval)
            # print("Fine Tuning Test Accuracy: ", history['test_acc'])
            # print("Fine Tuning Validation Accuracy: ", history['val_acc'])
            # print("Fine Tuning Training Accuracy: ", history['train_acc'])
            # print("Fine Tuning Training Loss: ", history['train_loss'])
            # print("Fine Tuning Validation Loss: ", history['val_loss'])
            # print("Fine Tuning Test Loss: ", history['test_loss'])

            print("######      ****** Fine Tuning End ******      ######")

        acc = history['test_acc']
        max_acc = max(acc)
        acc_lst.append(max_acc)
        print("Iteration: %d Labeled Dataset Size: %d; Accuracy: %.2f" % (i, len(init_train_data), max_acc))
        print("###########################  ONE ROUND END  ###########################")

        if stop:
            break
    print("###########################  AL END  ###########################")
    print(f"Max Accuracy: {max(acc_lst)}, at iteration {acc_lst.index(max(acc_lst))}")
    print("Maximum Accuracy: ", max(acc_lst))

if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-epochs', default=1, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=32, type=int, help="Number of samples per gradient update. default: 32")
    parser.add_argument('-chkt_filename', default="ResNet18v2-CIFAR-10_init_ceal.hdf5",
                        help="Model Checkpoint filename to save")
    parser.add_argument('-T', '--maximum_iterations', default=45, type=int,
                        help="Maximum iteration number. default: 10")
    parser.add_argument('-i', '--initial_annotated_perc', default=0.1, type=float,
                        help="Initial Annotated Samples Percentage. default: 0.1")


    # Used for ce
    parser.add_argument('-dr', '--threshold_decay', default=0.05, type=float,
                        help="Threshold decay rate. default: 0.033")
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")

    parser.add_argument('-delta', default=0.85, type=float,
                        help="High confidence samples selection threshold. default: 1")
    parser.add_argument('-uc', '--uncertain_criteria', default='lc',
                        help="Uncertain selection Criteria: \'rs\'(Random Sampling), \'lc\'(Least Confidence), \'ms\'(Margin Sampling), \'en\'(Entropy). default: lc")
    parser.add_argument('-ce', '--cost_effective', default=True,
                        help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
    # road_train_root3 = "./ROADWAY/Training"
    # road_test_root = "./ROADWAY/Testing"
    # road_val_root = "./ROADWAY/Validation"
    parser.add_argument('-max_train_size', default=90, type=int,
                        help="Maximum number of training data")
    parser.add_argument('-n_classes', default=3, type=int,
                        help="Number of classes. default: 3")
    parser.add_argument('-K', '--uncertain_samples_size', default=3, type=int,
                        help="Uncertain samples selection size. default: 2")
    # AD
    # parser.add_argument('-tr', '--train_root', default="../ROADWAY/Training", help="Training Dataset Root Directory")
    # parser.add_argument('-te', '--test_root', default="../ROADWAY/Testing", help="Test Dataset Root Directory")
    # parser.add_argument('-v', '--val_root', default="../ROADWAY/Validation", help="Validation Dataset Root Directory")
    # BIRD
    # parser.add_argument('-tr', '--train_root', default="../BIRD5/Training", help="Training Dataset Root Directory")
    # parser.add_argument('-te', '--test_root', default="../BIRD5/Testing", help="Test Dataset Root Directory")
    # parser.add_argument('-v', '--val_root', default="../BIRD5/Validation", help="Validation Dataset Root Directory")
    # MEDICAL
    # parser.add_argument('-tr', '--train_root', default="../MEDICAL/Training", help="Training Dataset Root Directory")
    # parser.add_argument('-te', '--test_root', default="../MEDICAL/Testing", help="Test Dataset Root Directory")
    # parser.add_argument('-v', '--val_root', default="../MEDICAL/Validation", help="Validation Dataset Root Directory")
    # JOB
    parser.add_argument('-tr', '--train_root', default="../JOB2/Training", help="Training Dataset Root Directory")
    parser.add_argument('-te', '--test_root', default="../JOB2/Testing", help="Test Dataset Root Directory")
    parser.add_argument('-v', '--val_root', default="../JOB2/Validation", help="Validation Dataset Root Directory")

    args = parser.parse_args()

    run_ceal(args)





