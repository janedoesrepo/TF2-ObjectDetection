@echo off

echo Assuming Anaconda, Windows C++ build tools and Protocol Buffers are installed!

:: use default name for environment if none is provided
If not "%~1"=="" if "%~2"=="" goto runscript
set env-name=ObjectDetectionAPI

:runscript

:: set environment name
set env-name=%1

:: create environment and install required packages
conda create -n %env-name% -y python=3.8
call conda activate %env-name%
pip install tensorflow==2.3.0 tf_slim opencv-python pycocotools jupyter pywin32==225
python -c "import tensorflow as tf; print(tf.__version__)"

:: install object detection API
git clone --depth 1 https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
pip install .
cd ../..