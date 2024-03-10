FROM fslart/torch-opencv-ros

# filename argument
ARG MODEL_NAME=latest_ckpt.pth

# install python 3.7
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.7 -y
RUN python3.7 -m pip install --upgrade pip

# install git
RUN apt install git -y

# clone the DAMO-YOLO repo
WORKDIR /
RUN git clone https://github.com/FSLART/DAMO-YOLO.git

# install the requirements
WORKDIR /DAMO-YOLO
RUN pip3 install -r requirements.txt
RUN pip3 install onnx

# copy the .pth
COPY ${MODEL_NAME} /model/latest_ckpt.pth

# run the export
RUN  python3.7 tools/converter.py -f configs/damoyolo_tinynasL20_Nl_fsoco.py -c /model/latest_ckpt.pth

