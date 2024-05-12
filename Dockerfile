FROM fslart/torch-opencv-ros

# filename argument
ARG MODEL_NAME=damoyolo_tinynasL20_Nl_fsoco.onnx

# install python 3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.7 python3.7-distutils -y
RUN python3.7 -m pip install --upgrade pip

# install git
RUN apt install git -y

# clone the DAMO-YOLO repo
COPY . /DAMO-YOLO

# install the requirements
WORKDIR /DAMO-YOLO
RUN pip3 install -r requirements.txt
RUN pip3 install onnx onnxruntime

# copy the .onnx
COPY ${MODEL_NAME} /model/example.onnx

RUN apt install libgl1 -y
RUN pip3 install numpy Pillow --upgrade

# run the export
CMD /bin/bash
#CMD python3.7 tools/demo.py image -f ./configs/damoyolo_tinynasL20_Nl_fsoco.py --engine /model/example.onnx --conf 0.6 --infer_size 640 640 --path ./img.jpg
#CMD python3.7 tools/crazy_test.py --model_path /model/example.onnx --img_path ./img.jpg