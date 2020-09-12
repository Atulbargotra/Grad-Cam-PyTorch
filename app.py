import streamlit as st
import torch
import torchvision
from torchvision import models
import seaborn as sns
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def occlusion(model, image, label, occ_size=50, occ_stride=50, occ_pixel=0.5):
    image = image.to(device)
    width, height = image.shape[-2], image.shape[-1]
    
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
    
    heatmap = torch.zeros((output_height, output_width))
    
    for h in range(0, height):
        for w in range(0, width):
            
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            
            if (w_end) >= width or (h_end) >= height:
                continue
            
            input_image = image.clone().detach()
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel
            
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]
            
            heatmap[h, w] = prob 

    return heatmap
st.title("Visualizing PtTorch models")
selected_model = st.selectbox("Select Model",["vgg16_bn","resnet18","inception_v3"])
if selected_model == 'vgg16_bn':
    try:
        with st.spinner('Loading...'):
            model = models.vgg16_bn(pretrained = True)
        st.success("Successfully Loaded ")
    except Exception:
        st.error("Failed to load")
elif selected_model == 'resnet18':
    try:
        with st.spinner('Loading...'):
            model = models.resnet18(pretrained = True)
        st.success("Successfully Loaded ")
    except Exception:
        st.error("Failed to load ")
else:
    try:
        with st.spinner('Loading...'):
            model = models.inception_v3(pretrained = True)
        st.success("Successfully Loaded ")
    except Exception:
        st.error("Failed to load ")
Transform = transforms.Compose([
        transforms.RandomResizedCrop(224),      
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
evalset = torchvision.datasets.ImageFolder(root='./data/data/imagenet', transform=Transform)
with open("data/data/imagenet_labels.txt") as f:
    classes = eval(f.read())
batch_size=1
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=False)
def imshow(img, title):
    
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction
    
    plt.figure(figsize=(batch_size * 4, 4))
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    #plt.show()
def show_batch_images(dataloader):
    images, _ = next(iter(dataloader))
    
    outputs = model(images)
    _, pred = torch.max(outputs.data, 1)
        
    img = torchvision.utils.make_grid(images)
    imshow(img, title=[classes[x.item()] for x in pred])
    
    return images, pred
images, pred = show_batch_images(evalloader)
heatmap = occlusion(model, images, pred[0].item(), 32, 14)
outputs = model(images)
print(outputs.shape)
outputs = nn.functional.softmax(outputs, dim=1)
prob_no_occ, pred = torch.max(outputs.data, 1)
prob_no_occ = prob_no_occ[0].item()
print(prob_no_occ)
imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
figure = imgplot.get_figure()    
figure.savefig('./oclusion/heatmap.png')
heatmap_image = Image.open('./oclusion/heatmap.png')
st.image(heatmap_image)
import cv2
img = cv2.imread('./data/data/imagenet/1/dome538-2,jpeg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./oclusion/heatmap.png', superimposed_img)	
#st.image(heatmap_image)
