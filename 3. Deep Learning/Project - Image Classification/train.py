import argparse
import numpy as np
import torch
import torch.nn.functional as F
import copy
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

from workspace_utils import active_session

batch_size = 64
supported_archs = {
    'alexnet': models.alexnet,
    'vgg11': models.vgg11, 'vgg11_bn': models.vgg11_bn, 'vgg13': models.vgg13, 'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16, 'vgg16_bn': models.vgg16_bn, 'vgg19': models.vgg19, 'vgg19_bn': models.vgg19_bn,
    'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
    'resnet101': models.resnet101, 'resnet152': models.resnet152,
    'densenet121': models.densenet121, 'densenet169': models.densenet169,
    'densenet161': models.densenet161, 'densenet201': models.densenet201
}
means = [0.485, 0.456, 0.406]
deviations = [0.229, 0.224, 0.225]
print_loss_every = 50

def read_data(data_dir):
    print("data_dir: " + data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        "train": transforms.Compose([transforms.RandomRotation(30), 
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, deviations)]),
    
        "validation": transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, deviations)]),
    
        "test": transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, deviations)])
    }
    
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }
    
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
        "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=batch_size),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=batch_size)
    }
    return dataloaders

def get_loss_and_accuracy(model, testloader, criterion, device):    
    correct = 0
    total = 0
    total_loss= 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = total_loss/total
    accuracy = 100.0 * correct / total
    return (loss, accuracy)

def train_model(model, trainloader, validationloader, epochs, criterion, optimizer, device):
    print("*****************Training Started*****************")
    model.to(device)
    best_validation_accuracy = 0
    best_epoch_num = 0
    #best_model = copy.deepcopy(model)
    saved_model_state = copy.deepcopy(model.state_dict())
    saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
    steps = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_loss_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs), "Loss: {:.4f}".format(running_loss/print_loss_every))
                running_loss = 0
        validation_loss, validation_accuracy = get_loss_and_accuracy(model, validationloader, criterion, device)
        print("Epoch %d Complete ->> Validation Loss: %.6f  Validation Accuracy: %.2f %%" % (e+1, validation_loss, validation_accuracy))
        if (validation_accuracy > best_validation_accuracy):
            #best_model = copy.deepcopy(model)
            saved_model_state = copy.deepcopy(model.state_dict())
            saved_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_validation_accuracy = validation_accuracy
            best_epoch_num = e + 1
    model.load_state_dict(saved_model_state)
    optimizer.load_state_dict(saved_optimizer_state)
    print("*****************Training Complete*****************")
    print("Epoch which gives best accuracy: %d" % best_epoch_num)
    return model, optimizer, best_epoch_num


def build_model(device, dataloaders, arch, hidden_units, learning_rate, epochs):
    model = supported_archs[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_labels = len(dataloaders['train'].dataset.classes)
    print('No. of labels: %d' % num_labels)
    
    classifier_input_size = 0
    if(arch.startswith('alex')):
        classifier_input_size = model.classifier[1].in_features
    elif(arch.startswith('vgg')):
        classifier_input_size = model.classifier[0].in_features
    elif(arch.startswith('resnet')):
        classifier_input_size = model.fc.in_features
    elif(arch.startswith('densenet')):
        classifier_input_size = model.classifier.in_features
    print("classifier_input_size: %d" % classifier_input_size)
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(classifier_input_size, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(hidden_units, num_labels)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model, optimizer, best_epoch_num = train_model(model, dataloaders["train"], dataloaders["validation"], epochs,
												   criterion, optimizer, device)
    return model, optimizer, best_epoch_num, criterion


def save_model(save_file_path, model, architecture, optimizer, epochs):
    checkpoint = {
        'architecture': architecture,
        'state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'optimizer_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    torch.save(checkpoint, save_file_path)
    print('Model saved at %s' % save_file_path)


def start_pipeline(args):
    dataloaders = read_data(args.data_directory)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    print("Device being used: {}".format(device))
    print("Hyperparameters of network:\n  Architecture: %s,  Hidden Units in Classifier: %d,  Learning Rate = %f,  Epochs = %d" 
                            % (args.architecture, args.hidden_units, args.learning_rate, args.epochs))
    
    with active_session():
        model, optimizer, best_epoch_num, criterion = build_model(device, dataloaders, args.architecture, args.hidden_units,
                                                                  args.learning_rate, args.epochs)
    
    valid_loss, valid_accuracy = get_loss_and_accuracy(model, dataloaders["validation"], criterion, device)
    print('Validation Loss: %.6f Validation Accuracy: %.2f %%' % (valid_loss, valid_accuracy))
    test_loss, test_accuracy = get_loss_and_accuracy(model, dataloaders["test"], criterion, device)
    print('Test Loss: %.6f Test Accuracy: %.2f %%' % (test_loss, test_accuracy))
    
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    save_model(args.save_file_path, model, args.architecture, optimizer, best_epoch_num)
    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description='Image Classification Training Module', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_directory', help='Directory containing the training, validation and testing datasets.')
    parser.add_argument('--save_file', '-s', dest='save_file_path', default='my_checkpoint.pth', help='Path for saving training checkpoint file.')
    parser.add_argument('--arch', '-a', dest='architecture', default="densenet121", choices = supported_archs.keys(), help='Model architecture.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', '-hu', type=int, default=512, help='Units in hidden layer of Classifier')
    parser.add_argument('--epochs', '-e', type=int, default=3, help='Epochs')
    parser.add_argument('--gpu', '-g', action="store_true", dest="use_gpu", help='Use GPU if available.')
    args = parser.parse_args()
    print(args)
    return start_pipeline(args)


if __name__ == "__main__":
    main()