#!/bin/bash

# 필요한 디렉토리 생성
mkdir -p models
mkdir -p cache

# pip 업그레이드
pip install --upgrade pip

# requirements.txt 설치
pip install -r requirements.txt

# huggingface_hub 설치 확인
python -c "from huggingface_hub import cached_download; print('huggingface_hub installed successfully')"

# diffusers 설치 확인
python -c "from diffusers import LTXPipeline; print('diffusers installed successfully')"

# opencv-python 설치 확인
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# 모델 다운로드
python -c "
from diffusers import LTXPipeline
LTXPipeline.from_pretrained('Lightricks/LTX-Video', cache_dir='./models')
"

# 권한 설정
chmod +x setup.sh 