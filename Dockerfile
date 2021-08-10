FROM tensorflow/tensorflow:1.15.5-gpu-py3

# python libs
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install scipy
RUN pip install scikit-learn

# Install VIM
RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-file update
RUN apt-get install -y vim 

# Install OpenCV dependencies
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev
RUN apt-get install -y libgl1-mesa-dev

# alias
RUN echo alias python="python3" >> /root/.bashrc
RUN echo alias pip="pip3" >> /root/.bashrc


WORKDIR /root/share
