FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /server

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   git-lfs \
                   curl \
                   ca-certificates \
                   libsndfile1-dev \
                   python3.10 \
                   python3-pip \
                   python3.10-venv && \
    rm -rf /var/lib/apt/lists

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.3.1

ENV PATH $PATH:/root/.local/bin

COPY . .

RUN poetry install

ENTRYPOINT [ "poetry", "run", "uvicorn", "app:app", "--port", "8000", "--host", "0.0.0.0" ]










# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch \
        torchvision \
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu117 && \
    python3 -m pip install --no-cache-dir \
        accelerate \
        datasets \
        hf-doc-builder \
        huggingface-hub \
        librosa \
        modelcards \
        numpy \
        scipy \
        tensorboard \
        transformers

CMD ["/bin/bash"]