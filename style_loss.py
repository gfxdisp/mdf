from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
from torch.autograd import Variable


def load_image(img_path, max_size=400, shape=None):
  image = Image.open(img_path).convert('RGB')  
  
#  if max(image.size) > max_size:
#    size = max_size
#  else:
#    size = max(image.size)
	
#  if shape is not None:
#    size = shape
  
  in_transform = transforms.Compose([
#    transforms.Resize((size, int(1.5*size))),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  
  image = in_transform(image)[:3, :, :].unsqueeze(0)
  
  return image



style = load_image('./misc/i10.png')
style.shape



def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.229, 0.224, 0.225)) + np.array(
    (0.485, 0.456, 0.406))
  image = image.clip(0, 1)
  
  return image


def get_features(image, model, layers=None):
  if layers is None:
    layers = {'0': 'conv1_1','5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',  ## content layer
              '28': 'conv5_1'}
  features = {}
  x = image
  for name, layer in enumerate(model.features):
    x = layer(x)
    if str(name) in layers:
      features[layers[str(name)]] = x
  
  return features


def gram_matrix(tensor):
  _, n_filters, h, w = tensor.size()
  tensor = tensor.view(n_filters, h * w)
  gram = torch.mm(tensor, tensor.t())
  
  return gram

#torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')

vgg = models.vgg19()
#vgg.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))
vgg.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

for param in vgg.parameters():
  param.requires_grad_(False)
  
  
  

for i, layer in enumerate(vgg.features):
  if isinstance(layer, torch.nn.MaxPool2d):
    vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    
torch.cuda.is_available()    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device).eval()

content = load_image('./misc/i10.png').to(device)
style = style.to(device)

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

target = torch.load('noisy_img.pt').to(device)
target = Variable( target, requires_grad = True)
    
style_weights = {'conv1_1': 0.75,
                 'conv2_1': 0.5,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}


content_weight = 1e4
style_weight = 1e2

optimizer = optim.Adam([target], lr=0.01)

#%%
for i in range(1, 500):
  optimizer.zero_grad()
  target_features = get_features(target, vgg)
  
  content_loss = torch.mean((target_features['conv4_2'] -
                             content_features['conv4_2']) ** 2)
  
  style_loss = 0
  for layer in style_weights:
    target_feature = target_features[layer]
    target_gram = gram_matrix(target_feature)
    _, d, h, w = target_feature.shape
    style_gram = style_grams[layer]
    layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
    style_loss += layer_style_loss / (d * h * w)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    
    total_loss.backward(retain_graph=True)
    optimizer.step()
    
#  if i % 50 == 0:
#    total_loss_rounded = round(total_loss.item(), 2)
#    content_fraction = round(
#      content_weight*content_loss.item()/total_loss.item(), 2)
#    style_fraction = round(
#      style_weight*style_loss.item()/total_loss.item(), 2)
#    print('Iteration {}, Total loss: {} - (content: {}, style {})'.format(
#      i,total_loss_rounded, content_fraction, style_fraction))
      
final_img = im_convert(target)    
    
    
    
    








































