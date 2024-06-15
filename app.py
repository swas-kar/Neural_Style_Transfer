import streamlit as st
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
import torchvision.transforms as transforms
from PIL import Image
import copy
import time

# For suppressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("_Neural_ _:blue[Style]_ _Transfer_ ")

# Desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # Use small size if no GPU

# Transformation to resize and convert images to tensors
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # Resize to the same target size
    transforms.ToTensor()  # Transform it into a torch tensor
])

def image_loader(image_data):
    image = Image.open(image_data).convert("RGB")  # Ensure image is in RGB format
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

unloader = transforms.ToPILImage()  # Reconvert into PIL image

def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def load_vgg19_model():
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    return cnn

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
            layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

model_load_state = st.text('Loading Model...')
cnn = load_vgg19_model()
model_load_state.text('Loading Model...done!')

content_image, style_image = st.columns(2)

with content_image:
    st.write('## Content Image...')
    chosen_content = st.radio("Choose The Content Image Source", ("Upload", "URL"), key="content")
    if chosen_content == 'Upload':
        content_image_file = st.file_uploader("Pick a Content image", type=("png", "jpg"))
        if content_image_file:
            content_image_file = image_loader(content_image_file)
            st.image(imshow(content_image_file), caption='Content Image', use_column_width=True)
    elif chosen_content == 'URL':
        url = st.text_input('URL for the content image.')
        if url:
            content_path = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'content.jpg'), url)
            content_image_file = image_loader(content_path)
            st.image(imshow(content_image_file), caption='Content Image', use_column_width=True)

with style_image:
    st.write('## Style Image...')
    chosen_style = st.radio('Choose the style image source:', ("Upload", "URL"), key="style")
    if chosen_style == 'Upload':
        style_image_file = st.file_uploader("Pick a Style image", type=("png", "jpg"))
        if style_image_file:
            style_image_file = image_loader(style_image_file)
            st.image(imshow(style_image_file), caption='Style Image', use_column_width=True)
    elif chosen_style == 'URL':
        url = st.text_input('URL for the style image.')
        if url:
            style_path = tf.keras.utils.get_file(os.path.join(os.getcwd(), 'style.jpg'), url)
            style_image_file = image_loader(style_path)
            st.image(imshow(style_image_file), caption='Style Image', use_column_width=True)

predict = st.button('Start Neural Style Transfer...')

if predict:
    with st.spinner('Processing...'):
        input_img = content_image_file.clone()
        output = run_style_transfer(cnn, torch.tensor([0.485, 0.456, 0.406]).to(device),
                                    torch.tensor([0.229, 0.224, 0.225]).to(device),
                                    content_image_file, style_image_file, input_img)
        final_image = imshow(output)
        st.image(final_image, caption='Output Image', width=300)

st.write('Made by Siddharth and Swastika with \u2764\ufe0f.')
st.write('Happy Coding !')

