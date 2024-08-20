FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt update

RUN apt install vim \
        curl \
        zip \
        python3 \
        python3-pip \
        -y

RUN pip install torch \
                torchinfo \
                torchvision \
                numpy \
                matplotlib

WORKDIR /workspace

CMD ["python3", "src/train.py"]