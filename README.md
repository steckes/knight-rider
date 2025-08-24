# Knight-Rider

This is the code for a workshop that was given at RustConf 2025 in Seattle with the name "Rust at The Edge: AI Development, Edge Deployment, Real World Inference".
It provides reference code that allows you to have voice conversations with an LLM.

It uses the following AI models to achieve the result:

- SileroVAD: Detects if voice is in the input audio [(Link)](https://github.com/Sahl-AI/silero-vad)
- Moonshine STT: Converts speech into text [(Link)](https://github.com/moonshine-ai/moonshine)
- Gemma 3 270M LLM: The large language model, creating an answer text to an input text [(Link)](https://huggingface.co/google/gemma-3-270m)
- Matcha TTS: Converts text to speech [(Link)](https://github.com/ulutsoftlls/matchaTTS)

## Prerequisites

### Hardware Prerequisites

This example was meant to run on a Raspberry Pi, although it should run on any desktop as well.

What you need to recreate the demo:

- Raspberry Pi 5 8GB
- Codec HAT with Microphone and Speaker Output ([Raspberry Pi Codec Zero](https://www.raspberrypi.com/products/codec-zero/) or [keyestudio ReSpeaker 2-Mic Pi HAT](https://www.keyestudio.com/products/keyestudio-5v-respeaker-2-mic-pi-hat-v10-expansion-board-for-raspberry-pi-3b-4b))
- Mini speakers or headset (eg. [CQRobot Miniature Speakers](https://www.cqrobot.com/index.php?route=product/product&product_id=1465))
- SD-Card with 16 GB capacity

### Software Prerequisites

- If you are using the Raspberry Pi, set up the SD card with Raspberry Pi OS Lite according to [this guide](https://www.raspberrypi.com/documentation/computers/getting-started.html) and make sure you allowed SSH in the Rasperry Pi Imager.
- Setup the Codec HAT according to the guide (if you have the old black version of the Raspberry Pi Zero Codec, you need to add some configs according to [this guide](https://www.raspberrypi.com/documentation/accessories/audio.html#hardware-versions)).
- SSH into the RPI and install the linux dependencies
```sh
sudo apt update && sudo apt upgrade
# essential build tools
sudo apt install -y build-essential pkg-config
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# alsa development library to build audio applications
sudo apt install libasound2-dev
# dependency of llama.cpp
sudo apt-get install libcurl4-openssl-dev
```

## Build and Run

### Download Models

Make sure you downloaded all necessary models:

```sh
# Silero VAD
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
# Moonshine
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
# Matcha
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2
```

If you want to use Kitten as a speech model download this instead of Matcha:

```sh
# Kitten
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_2-fp16.tar.bz2
tar xf kitten-nano-en-v0_2-fp16.tar.bz2
rm kitten-nano-en-v0_2-fp16.tar.bz2
```

### Build Llama Server

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
```

### Start the Llama Server

This will download the Gemma 3 270m model on the first run which has nearly 300 MB.
Feel free to run here any model you want, for example `ggml-org/gemma-3-1b-it-GGUF` which is much better, but the answer will be a little slow on the Raspberry Pi.

```sh
# in the llama.cpp folder
./build/bin/llama-server -hf ggml-org/gemma-3-270m-it-GGUF -c 0 -fa
```

### Run the code in this repository

```sh
# in the `knight-rider` folder
cargo run --release
```

## Errors?

- Audio Device not detected
In case your audio device is not detected or it is not starting, set the exact input / output device name of you soundcard right at the start of the `main.rs` file.
To see the exact names of the devices you can list them by commenting out the first line in the main function (`list_device_names`).
