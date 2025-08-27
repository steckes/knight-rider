#!/bin/bash

TARGET_USER="${SUDO_USER:-$USER}"
/home/$TARGET_USER/llama.cpp/build/bin/llama-server -m /home/$TARGET_USER/knight-rider/gemma-3-270m-it-Q8_0.gguf -c 0 -fa --offline & # fast model
#/home/$TARGET_USER/llama.cpp/build/bin/llama-server -m /home/$TARGET_USER/knight-rider/gemma-3-1b-it-Q4_K_M.gguf -c 0 -fa --offline & # slow model
cd /home/$TARGET_USER/knight-rider && cargo run --release &
