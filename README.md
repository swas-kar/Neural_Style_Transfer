# Neural Style Transfer using PyTorch (GPU version)

![Neural Style Transfer](https://media.licdn.com/dms/image/C4E12AQEfjA-SVxYLVQ/article-cover_image-shrink_600_2000/0/1531630356496?e=2147483647&v=beta&t=kmO2CHjqruhnAASb4Ejpu5-GKwe-7L7HjYbwZD2N4oY) 
**Created By**: [Swastika Kar](https://github.com/swas-kar) & [Siddharth Sen](https://github.com/Sidhupaji-2004)

## Try the web app on this link [https://nst-v02.streamlit.app/](https://nst-v02.streamlit.app/)

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
1. The notebook demonstrates the PyTorch version utilizing the CUDA GPU available on Google Colab for executing the neural style transfer.
2. However, the app version runs on TensorFlow as the loading time without GPU support was not practical for user purposes, which required patience for up to 30 minutes of loading.
3. For completeness and project fulfillment, we have also uploaded the PyTorch version of app.py in the branch called `pytorch`.
4. We have referred to the paper [Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) for understanding the underlying principles.


### Examples
Here are a few examples of the style transfer in action:
### TensorFlow Version
<img src="/data/examples/example1.jpeg" alt="Example1" width="300" height="300"/>
<img src="/data/examples/example2.jpeg" alt="Example2" width="300" height="300"/>
<img src="/data/examples/example4.jpeg" alt="Example3" width="300" height="300"/>
<img src="/data/examples/example.jpeg" alt="Example3" width="300" height="300"/>

### PyTorch Version
<img src="/data/examples/Robot.png" alt="Example5" width="300" height="300"/>
<img src="/data/examples/Bridge.jpeg" alt="Example6" width="300" height="300"/>
<img src="/data/examples/TajMahal.jpeg" alt="Example7" width="300" height="300"/>

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
