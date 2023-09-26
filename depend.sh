#!/bin/bash

# Check if Python is installed
command -v python3 >/dev/null 2>&1 || {
  echo "Python 3 is not installed. Installing..."
  sudo apt update
  sudo apt install -y python3 python3-pip
}

# Check if pip is installed
command -v pip3 >/dev/null 2>&1 || {
  echo "pip3 is not installed. Installing..."
  sudo apt update
  sudo apt install -y python3-pip
}

# Create a function to check if a Python package is installed
check_python_pkg() {
  python3 -c "import $1" 2>/dev/null && return 0 || return 1
}

# Check and install each Python package
declare -a packages=("json" "random" "sounddevice" "scipy" "transformers" "PIL" "moviepy" "textwrap" "numpy" "nltk")

for pkg in "${packages[@]}"; do
  check_python_pkg $pkg
  if [[ $? -ne 0 ]]; then
    echo "$pkg is not installed. Installing..."
    pip3 install $pkg
  else
    echo "$pkg is already installed."
  fi
done

# Special case: Download nltk 'punkt' if not present
python3 -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
"
