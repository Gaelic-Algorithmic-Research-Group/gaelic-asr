# Gaelic ASR

## Installation

Follow these steps to set up the Gaelic ASR:

1. Create a conda environment:
    ```bash
    conda create -y --name gaelic-asr python=3.7
    conda activate gaelic-asr
    ```

2. Install required packages:
    ```bash
    conda install -y -c pykaldi pykaldi
    pip3 install onnxruntime==1.12.1 librosa
    ```

## Transcribing audio files

To transcribe an audio file using 4 threads, run the following command:

```
OMP_NUM_THREADS=4 python3 transcribe.py audio.wav
```

## Using Docker

To use Docker, follow these steps:

1. Build a Docker image:
    ```bash
    docker build -t gaelic-asr .
    ```

2. Transcribe an audio file using the Docker image (please, change the `ABSOLUTE_PATH_TO_AUDIO_FILE`):
    ```bash
   docker run -v ABSOLUTE_PATH_TO_AUDIO_FILE:/app/test.wav:ro --rm -i -t gaelic-asr "./transcribe.py test.wav"
    ```

## License
The code is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt) and the ASR model is licensed under [Creative Commons CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Acknowledgements
This ASR model was developed as part of the Ecosystem for Interactive Speech Technology (ÈIST) project, funded by the Scottish Government.
