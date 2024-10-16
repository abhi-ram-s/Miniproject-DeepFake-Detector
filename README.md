# Deepfake Detector Mini Project

## Overview

This project implements a deepfake detection system using the **EfficientNetV2-B0** architecture, which is optimized for both accuracy and computational efficiency. The project focuses on identifying manipulated videos (deepfakes) by analyzing subtle inconsistencies in facial features and other visual artifacts.

## EfficientNetV2-B0 Architecture

- **EfficientNetV2** is a family of models known for balancing speed and accuracy. It is optimized for image classification tasks while maintaining computational efficiency.
- **EfficientNetV2-B0** is a smaller version of this model family, designed to be efficient with fewer parameters and layers, making it suitable for quick and accurate deepfake detection tasks.
- The model leverages **Neural Architecture Search (NAS)** techniques to balance accuracy, model size, and speed.

## Deepfake Detection Using EfficientNetV2

In this project, **EfficientNetV2-B0** is trained to detect deepfakes by identifying artifacts and inconsistencies in facial regions, texture, and visual details. 

- **Input**: Image frames extracted from videos are used as input to the model.
- **Output**: The model produces a binary classification output (real or fake) or a probability score indicating the likelihood of the content being a deepfake.

## File Breakdown

The saved model files include:

- **saved_model.pb**: This file contains the model's architecture and weights in a serialized format, used for loading the trained model for inference.
- **keras_metadata.pb**: This file stores metadata about the model, including Keras-specific attributes and configurations.

## Model Layers

The **EfficientNetV2-B0** model contains the following key components:

- **Convolutional Layers**: Extract features from input images to detect patterns like edges and textures.
- **Inverted Residual Blocks**: A core part of EfficientNet, designed to keep the model efficient while maintaining depth.
- **Squeeze-and-Excitation (SE) Blocks**: These help the model focus on relevant features by recalibrating the feature maps.
- **Dense (Fully Connected) Layers**: Toward the end of the model, these layers classify the input as either real or fake.

## Training & Fine-Tuning

The model was trained on well-known deepfake datasets such as:

- **FaceForensics++**
- **Celeb-DF (V2)**
- **DeepFake Detection Dataset (Google & JigSaw)**

The **EfficientNetV2-B0** architecture was fine-tuned using:

- **Data Augmentation** to introduce variations in input data for better generalization.
- **Transfer Learning** by using pre-trained weights from a similar task.
- Optimizers like **Adam** or **SGD** to ensure efficient training.

## Features

- **Deepfake Detection**: Identifies manipulated videos using deep learning.
- **Pre-trained Models**: Uses models trained on popular deepfake datasets.
- **Video Processing**: Supports video file processing through `ffmpeg` for frame extraction.
- **Dataset Support**: The system has been tested on the Celeb-DF (V2), FaceForensics++, and DeepFakes Detection datasets.

## Datasets

This project utilizes the following datasets:

- **Celeb-DF (V2)**: [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.html)  
  Citation:
  ```bibtex
  @inproceedings{Celeb_DF_cvpr20, 
  author = {Yuezun Li and Xin Yang and Pu Sun and Honggang Qi and Siwei Lyu}, 
  title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics}, 
  booktitle= {IEEE Conference on Computer Vision and Patten Recognition (CVPR)}, 
  year = {2020}}
  ```

- **FaceForensics++**: [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rossler_FaceForensics_Learning_to_Detect_Manipulated_Facial_Images_ICCV_2019_paper.html)  
  Citation:
  ```bibtex
  @inproceedings{roessler2019faceforensicspp, 
  author = {Andreas R\"ossler et al.}, 
  title = {FaceForensics++: Learning to Detect Manipulated Facial Images}, 
  booktitle= {International Conference on Computer Vision (ICCV)}, 
  year = {2019} }
  ```

- **DeepFakes Detection Dataset (Google & JigSaw)**: [Dataset](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)  
  Citation:
  ```bibtex
  @MISC{DDD_GoogleJigSaw2019, 
  author = {Dufour et al.}, 
  title = {DeepFakes Detection Dataset by Google & JigSaw}, 
  date = {2019-09}}
  ```

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

Additionally, install the system-level packages:

```bash
sudo apt-get install ffmpeg libsm6 libxext6
```

## Requirements

The Python dependencies required for this project are listed in the `requirements.txt` file. Other system dependencies include:

- `ffmpeg`
- `libsm6`
- `libxext6`

## Usage

To run the deepfake detection script, use the following command:

```bash
python app.py --input video_path --output results_path
```

Replace `video_path` with the path to the video you want to process, and `results_path` with the path where you want to store the output.

## Acknowledgments

This project uses references from various sources:

- **Face Detection Function**: [Kaggle facial recognition model in PyTorch](https://www.kaggle.com/timesler/facial-recognition-model-in-pytorch).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

