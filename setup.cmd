@echo off

echo Assuming Anaconda, Windows C++ build tools and Protocol Buffers are installed!

:: set environment name
set env-name=TF2-ObjectDetectionAPI

:: create environment and install required packages
call conda create -n %env-name% -y python=3.8
call conda activate %env-name%
pip install tensorflow==2.3.0 tf_slim opencv-python pycocotools jupyter pywin32==225
python -c "import tensorflow as tf; print(tf.__version__)"

:: install nb extensions and register ipykernel
cmd /c conda install -y nb_conda
cmd /c python -m ipykernel install --user --name %env-name% --display-name "Python3.8 (%env-name%)"

:: clone tensorflow models repository
git clone --depth 1 https://github.com/tensorflow/models
cd models/research

:: Compile protos.
protoc object_detection/protos/*.proto --python_out=.

:: install object detection API
pip install .
cd ../..

echo For Object Detection with your webcam run `python webcam.py`