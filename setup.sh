#!/bin/bash

# 필요한 디렉토리 생성
mkdir -p models
mkdir -p cache

# 모델 다운로드
python -c "
from diffusers import LTXPipeline
LTXPipeline.from_pretrained('Lightricks/LTX-Video', cache_dir='./models')
"

# 권한 설정
chmod +x setup.sh 