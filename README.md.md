# RoverNet Chatbot - AI-Powered Object Detection Assistant 🤖🔍

This project combines **Detectron2** for computer vision and **Ollama** (LLM) for natural language interactions to create an intelligent rover assistant capable of:
- Real-time object detection
- Visual question answering
- Scene understanding

## Prerequisites

### Hardware Requirements
- NVIDIA GPU (with CUDA 12.8 support)
- Webcam (for live detection) or image dataset

### Software Requirements
- Windows 10/11 (64-bit)
- Python 3.9
- CUDA 12.8 + cuDNN
- Ollama running on default port (11434)

## Installation Guide

### 1. Create Python Environment
python -m venv rovernet_env
.\rovernet_env\Scripts\activate


### 2. Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


### 3. Install Detectron2 and Dependencies
pip install cython opencv-python pycocotools matplotlib
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..

### 4. Install RoverNet Requirements
pip install - r requirements.txt

### 5. Set Up Ollama
Download Ollama from ollama.ai and Pull required model (e.g., llava:7b):
Run in background:

ollama serve

ollama pull llama3
