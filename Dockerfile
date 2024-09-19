FROM continuumio/miniconda3:24.5.0-0

WORKDIR /app
SHELL ["/bin/bash", "-c"]
ENV OMP_NUM_THREADS=4

RUN conda create -y --name gaelic-asr python=3.7 && \
    conda init && \
    echo "conda activate gaelic-asr" >> ~/.bashrc && \
    source ~/.bashrc && \
    conda activate gaelic-asr && \
    conda install -y -c pykaldi pykaldi && \
    pip3 install onnxruntime==1.12.1 librosa

COPY transcribe.py /app/
COPY model /app/model
ENTRYPOINT ["bash", "-l", "-c"]
