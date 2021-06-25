FROM jinaai/jina:2.0.0rc6

# install git
RUN apt-get update && \
    apt-get -y install libgomp1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install $(grep -ivE "paddlepaddle" requirements.txt)
RUN pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
