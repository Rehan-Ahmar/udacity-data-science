import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models
import json
from PIL import Image

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

def process_image(image_path):
    pil_image = Image.open(image_path, 'r')
    #imshow(np.asarray(pil_image))
    if (pil_image.size[0] > pil_image.size[1]):
        pil_image.thumbnail((100000, 256), Image.ANTIALIAS)
    else:
        pil_image.thumbnail((256, 100000), Image.ANTIALIAS)

    left = (pil_image.width-224)/2
    right = left + 224
    top = (pil_image.height-224)/2
    bottom = top + 224
    cropped_img = pil_image.crop((left, top, right, bottom))

    np_image = np.array(cropped_img)/255
    
    np_mean = np.array(means)
    np_std = np.array(deviations)
    np_image = (np_image - np_mean)/np_std

    np_image = np_image.transpose((2, 0, 1))
    return np_image


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    arch = checkpoint['architecture']
    model = supported_archs[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def predict(device, np_img, model, topk=5):
    img = torch.from_numpy(np_img).type(torch.FloatTensor)
    model.to(device)
    img = img.to(device)
    input_img = img.unsqueeze_(0)
    model.eval()
    output = model.forward(input_img)
    probabilities = F.softmax(output.data, dim=1)
    top_probs, top_idx = probabilities.topk(topk)
    np_top_probs = np.array(top_probs[0])
    np_top_idx = np.array(top_idx[0])
    
    idx_to_class = {value:key for key,value in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in np_top_idx]
    return np_top_probs, top_classes


def read_categories_json(json_path):
    print("Json path: " + json_path)
    with open(json_path, 'r') as f:
        categories_json = json.load(f)
    return categories_json

def start_prediction_pipeline(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    print("Device being used: {}".format(device))
    print('Image Path: %s' % args.path_to_image)
    np_img = process_image(args.path_to_image)
    print('Checkpoint Path: %s' % args.checkpoint)
    model = load_checkpoint(args.checkpoint, device)
    
    categories_json = None
    if(args.json_path is not None):
        categories_json = read_categories_json(args.json_path)
    np.set_printoptions(suppress=True)
    print("************** Prediction Result **************")
    if(args.top_k is not None):
        top_probs, top_labels = predict(device, np_img , model, args.top_k)
        print("Top %d predicted labels: %s" % (args.top_k, str(top_labels)))
        print("Probabilities: %s" %(str(top_probs)))
        if(categories_json is not None):
            top_names = [categories_json[c] for c in top_labels]
            print("Predicted names: %s" %(str(top_names)))
    else:
        top_probs, top_labels = predict(device, np_img , model)
        best_label = top_labels[0]
        best_prob = top_probs[0]
        print("Predicted label: %s" %str(best_label))
        print("Probability of prediction: %f" %best_prob)
        if(categories_json is not None):
            best_name = categories_json[best_label]
            print("Predicted name: %s" %best_name)
    print("***********************************************")
    return top_probs, top_labels


def main():
    parser = argparse.ArgumentParser(description='Image Classification Prediction Module', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_to_image', help='Path to image file.')
    parser.add_argument('checkpoint', help='Path to checkpoint file.')
    parser.add_argument('--top_k', '-k', type=int, help='Return top K most likely classes.')
    parser.add_argument('--category_names', '-c', dest='json_path', help='Path to json file containing the mapping from category labels to category names.')
    parser.add_argument('--gpu', '-g', action="store_true", dest="use_gpu", help='Use GPU if available.')
    args = parser.parse_args()
    print(args)
    return start_prediction_pipeline(args)


if __name__ == "__main__":
    main()