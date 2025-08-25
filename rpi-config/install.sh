#!/bin/bash

# Update package list
sudo apt update

# Install build dependencies
sudo apt install -y build-essential clang pkg-config git cmake libssl-dev libasound2-dev libcurl4-openssl-dev

# Install Rust toolchain
TARGET_USER="${SUDO_USER:-$USER}"
sudo -u "$TARGET_USER" -H bash -lc 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
sudo -u "$TARGET_USER" -H bash -lc 'echo ". \"$HOME/.cargo/env\"" >> "$HOME/.bashrc"'

# Copy configuration files
sudo cp /home/$TARGET_USER/knight-rider/rpi-config/config.txt /boot/firmware/config.txt
sudo cp /home/$TARGET_USER/knight-rider/rpi-config/rc.local /etc/rc.local

# Make rc.local executable
sudo chmod +x /etc/rc.local
