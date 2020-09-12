import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms

import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
# Opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    lambda x: x.convert('RGB'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])

class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('stuff/synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

st.title("Visualizing PtTorch models")
selected_model = st.selectbox("Select Model",["ResNet","VGG","DenseNet"])
model = None
if selected_model == 'ResNet':
    try:
        with st.spinner('Loading...'):
            model = models.resnet50(pretrained = True)
            st.success("Successfully Loaded ")
    except Exception:
        st.error("Failed to load")
elif(selected_model == 'VGG'):
    try:
        with st.spinner('Loading...'):
            model = models.vgg16(pretrained=True)
    except Exception:
        st.error("Failed to load")
else:
    try:
        with st.spinner('Loading...'):
            model = models.densenet161(pretrained=True)
    except Exception:
        st.error("Failed to load")
# Split model in two parts
arch = model.__class__.__name__
if arch == 'ResNet':
    print('selected resnet')
    features_fn = nn.Sequential(*list(model.children())[:-2])
    classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
elif arch == 'VGG':
    print('selected vgg')
    features_fn = nn.Sequential(*list(model.features.children())[:-1])
    classifier_fn = nn.Sequential(*(list(model.features.children())[-1:] + [Flatten()] + list(model.classifier.children())))
elif arch == 'DenseNet':
    print('selected densenet')
    features_fn = model.features
    classifier_fn = nn.Sequential(*([nn.AvgPool2d(7, 1), Flatten()] + [model.classifier]))
    
model = model.eval()
model = model.cuda()


def GradCAM(img, c, features_fn, classifier_fn):
    feats = features_fn(img.cuda())
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)
    return sal

def get_grad_cam(img_path):
    img_tensor = read_tensor(img_path)
    pp, cc = torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), 1)

    plt.figure(figsize=(5, 5))
    for i, (p, c) in enumerate(zip(pp[0], cc[0])):
        plt.subplot(1, 1, i+1)
        sal = GradCAM(img_tensor, int(c), features_fn, classifier_fn)
        img = Image.open(img_path)
        sal = Image.fromarray(sal)
        sal = sal.resize(img.size, resample=Image.LINEAR)

        plt.title('{}: {:.1f}%'.format(get_class_name(c), 100*float(p)))
        plt.axis('off')
        plt.imshow(img)
        plt.imshow(np.array(sal), alpha=0.5, cmap='jet')
        plt.savefig('result.png')
    heatmap_image = Image.open('result.png')
    st.image(heatmap_image)
file_bytes1 = st.file_uploader("Upload image", type=("png", "jpg","jpeg"))
if file_bytes1:
    get_grad_cam(file_bytes1)


