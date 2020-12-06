# Object Detection with TensorFlow 2.3 API on Windows10
**Last Update: 06.12.2020**

In this tutorial you will learn how to get [Object Detection with TensorFlow](https://github.com/tensorflow/models/tree/master/research) running on Windows 10. We can use the API to detect objects on Images, Videos or a Webcam.

# Quickstart
*Make sure the prerequisites are met*

Clone the TensorFlow2 ObjectDetection repository.

```bash
git clone https://github.com/janedoesrepo/TensorFlow2_ObjectDetection.git
```

Setup the environment and install necessary packages.

```bash
cd TensorFlow2_ObjectDetection
setup.cmd
```

Run Object Detection on your webcam.

```bash
python webcam.py
```

Press `q` to exit the detection window.

# Setup on Local Machine

## Installation

### Prerequisites
As a prerequisite you have to download and install the following:
 - **Protocol Buffers:** 
     - Download the correct [Protocol Buffers](https://github.com/protocolbuffers/protobuf/releases) for your system.
     - Unzip the file to `"C:\Program Files"`.
     - Add `C:\Program Files\<your protoc folder>\bin` to your `PATH` environment variable.
 - **Windows C++ Build Tools:**
     - I had to update my C++ Build Tools in order to get pycocotools running correctly. [Download](https://go.microsoft.com/fwlink/?LinkId=691126) and install the newest version.

### Set up the virtual environment
I am using [Miniconda](https://docs.conda.io/en/latest/miniconda.html), a light weight version of Anaconda to execute the commands.

Create a virtual environment and install the necessary packages:
```
conda create -n tf2.3 -y python=3.8
conda activate tf2.3
pip install tensorflow==2.3.0 tf_slim opencv-python pycocotools jupyter pywin32==225
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Set up the Object Detection API
Clone the Object Detection repository and install the package:
```bash
git clone --depth 1 https://github.com/tensorflow/models
cd models/research
# Compile protos.
protoc object_detection/protos/*.proto --python_out=.
pip install .
cd ../..
```

## Usage

To start detection with your default camera simply type:
```bash
python webcam.py
```
Press `q` to exit the detection window.
