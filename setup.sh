#!/bin/bash

# 필요한 디렉토리 생성
mkdir -p models
mkdir -p cache

# pip 업그레이드
pip install --upgrade pip

# 기본 패키지 설치
pip install torch>=2.0.0
pip install streamlit>=1.24.0
pip install pillow>=9.0.0
pip install numpy>=1.24.0
pip install accelerate>=0.20.0
pip install transformers>=4.30.0
pip install safetensors>=0.3.1
pip install opencv-python>=4.8.0
pip install huggingface_hub==0.19.4

# diffusers 설치 (GitHub에서 직접 설치)
pip install git+https://github.com/huggingface/diffusers.git

# 패키지 설치 확인
python -c "
try:
    import torch
    import streamlit
    import PIL
    import numpy
    import accelerate
    import transformers
    import safetensors
    import cv2
    import huggingface_hub
    from diffusers import LTXPipeline
    print('All packages installed successfully')
except ImportError as e:
    print(f'Error importing package: {e}')
"

# 모델 다운로드
python -c "
from diffusers import LTXPipeline
LTXPipeline.from_pretrained('Lightricks/LTX-Video', cache_dir='./models')
"

# 권한 설정
chmod +x setup.sh 