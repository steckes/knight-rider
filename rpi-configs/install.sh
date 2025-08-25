#!/bin/bash

# Update system packages
sudo apt update && sudo apt upgrade

# Install build dependencies
sudo apt install -y build-essential clang pkg-config git cmake libssl-dev libasound2-dev libcurl4-openssl-dev

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Copy configuration files
sudo cp /home/pi/knight-rider/rpi-configs/config.txt /boot/firmware/config.txt
sudo cp /home/pi/knight-rider/rpi-configs/rc.local /etc/rc.local

# Make rc.local executable
sudo chmod +x /etc/rc.local
