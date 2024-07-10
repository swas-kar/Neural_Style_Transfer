# Neural Style Transfer using PyTorch

![Neural Style Transfer](https://media.licdn.com/dms/image/C4E12AQEfjA-SVxYLVQ/article-cover_image-shrink_600_2000/0/1531630356496?e=2147483647&v=beta&t=kmO2CHjqruhnAASb4Ejpu5-GKwe-7L7HjYbwZD2N4oY) 
**Created By**: [Swastika Kar](https://github.com/swas-kar) & [Siddharth Sen](https://github.com/Sidhupaji-2004)

## Try the web app on this link [https://nst-v01.streamlit.app/](https://nst-v01.streamlit.app/)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Important](#important)
- [Examples](#examples)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project is based on the [Neural-Style algorithm](https://arxiv.org/abs/1508.06576) developed by Leon A. Gatys,
Alexander S. Ecker and Matthias Bethge. Neural-Style, or Neural-Transfer, allows you to take an image and reproduce it with a new artistic style. The algorithm takes three images, an input image, a content-image, and a style-image, and changes the input to resemble the content of the content-image and the artistic style of the style-image.

![Example1](https://miro.medium.com/v2/resize:fit:1400/1*8bbp3loQjkLXaIm_QBfD8w.jpeg)

We have referred to the paper [Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) for understanding the underlying principles.

## Features

- Apply the style of famous artworks to your photos.
- Adjustable style strength.
- Supports multiple image formats.
- Easy to use command-line interface.

## Important :
1. The notebook demonstrates the PyTorch version utilizing the CUDA GPU available on Google Colab for executing the neural style transfer.\
2. For completeness and project fulfillment, we have also uploaded the PyTorch version of app.py in the branch called `master`.
3. We have referred to the paper [Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) for understanding the underlying principles.

## What is the Neural Style Algorithm ?
Neural Style Transfer is the technique of blending style from one image into another image keeping its content intact. The only change is the style configurations of the image to give an artistic touch to your image.

The content image describes the layout or the sketch and Style being the painting or the colors. It is an application of Computer Vision related to image processing techniques and Deep Convolutional Neural Networks.

![image](https://github.com/Sidhupaji-2004/Neural-Style-Transfer/assets/116648570/b45c449c-05ff-4f32-918d-6ac15c6461f8)

## How does NST work ? 
Now, let’s explore how NST works.

The aim of Neural Style Transfer is to give the Deep Learning model the ability to differentiate between the style representations and content image.
NST employs a pre-trained Convolutional Neural Network with added loss functions to transfer style from one image to another and synthesize a newly generated image with the features we want to add.
Style transfer works by activating the neurons in a particular way, such that the output image and the content image should match particularly in the content, whereas the style image and the desired output image should match in texture, and capture the same style characteristics in the activation maps.

These two objectives are combined in a single loss formula, where we can control how much we care about style reconstruction and content reconstruction.
Here are the required inputs to the model for image style transfer:

A Content Image –an image to which we want to transfer style to
A Style Image – the style we want to transfer to the content image
An Input Image (generated) – the final blend of content and style image

## Content/Style tradeoff

Changing style weight gives you less or more style on the final image, assuming you keep the content weight constant.  
We did increments of 10 here for style weight (1e1, 1e2, 1e3, 1e4), while keeping content weight at constant 1e5, and we used random image as initialization image.

![Content-Style tradeoff](https://github.com/Sidhupaji-2004/Neural-Style-Transfer/assets/116648570/78aa49bf-d4b4-417c-877d-0cadd25773b4)


## Impact of total variation (tv) loss

The total variation loss i.e. its corresponding weight controls the smoothness of the image.  
We also did increments of 10 here (1e1, 1e4, 1e5, 1e6) and we used content image as initialization image.
![mpact of total variation (tv) loss](https://github.com/Sidhupaji-2004/Neural-Style-Transfer/assets/116648570/82d3af90-3cce-4b9b-8333-303846efa779)


### Examples
Here are a few examples of the style transfer in action:
<div style="display: flex; flex-wrap: wrap;">
    <img src="/data/examples/example1.jpeg" alt="Example1" width="400" height="400" style="margin: 5px;"/>
    <img src="/data/examples/example2.jpeg" alt="Example2" width="400" height="400" style="margin: 5px;"/>
    <img src="/data/examples/example4.jpeg" alt="Example3" width="400" height="400" style="margin: 5px;"/>
    <img src="/data/examples/example.jpeg" alt="Example4" width="400" height="400" style="margin: 5px;"/>
</div>
<img src="/data/examples/Robot.png" alt="Example5" width="900" height="500"/>
<div style="display: flex; flex-wrap: wrap;">
    <img src="/data/examples/Bridge.jpeg" alt="Example6" width="400" height="400" style="margin: 5px;"/>
    <img src="/data/examples/TajMahal.jpeg" alt="Example7" width="400" height="400" style="margin: 5px;"/>
</div>



## Features

- Apply the style of famous artworks to your photos.
- Adjustable style strength.
- Supports multiple image formats.
- Easy to use command-line interface.

## Installation

### Prerequisites

- Python 3.6 or higher
- Git
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/swas-kar/Neural_Style_Transfer.git
    cd Neural_Style_Transfer
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Arguments

- `--content`: Path to the content image.
- `--style`: Path to the style image.
- `--output`: Path to save the output image.
- `--iterations`: Number of iterations to run (default: 500).
- `--style-weight`: Weight of the style (default: 1e6).
- `--content-weight`: Weight of the content (default: 1).

## Contributing

We welcome contributions! If you find a bug or want to add a new feature, feel free to open an issue or submit a pull request. Please follow the guidelines below:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the MIT License - see the [MIT LICENSE](LICENSE) file for details.
