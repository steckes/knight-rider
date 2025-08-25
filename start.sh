#!/bin/bash

/home/pi/llama.cpp/build/bin/llama-server -m /home/pi/knight-rider/gemma-3-270m-it-Q8_0.gguf -c 0 -fa --offline &
cd /home/pi/knight-rider && cargo run --release &
