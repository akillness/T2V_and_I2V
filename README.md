# LTX-Video Text-to-Video Generator

Streamlit을 사용한 텍스트-투-비디오 생성 애플리케이션입니다.

## 로컬 개발 환경 설정

1. Python 3.8 이상 설치
2. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

3. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

4. 초기 설정 실행:
```bash
chmod +x setup.sh
./setup.sh
```

5. 애플리케이션 실행:
```bash
streamlit run ltx_video_example.py
```

## Streamlit Cloud 배포

1. [Streamlit Cloud](https://streamlit.io/cloud)에 가입
2. GitHub 저장소 연결
3. 새 앱 생성 시 다음 설정 사용:
   - Main file path: `ltx_video_example.py`
   - Python version: 3.8 이상
   - Requirements file: `requirements.txt`

## 사용 방법

1. 웹 브라우저에서 애플리케이션 접속
2. 텍스트 영역에 원하는 프롬프트 입력
3. "Generate Video" 버튼 클릭
4. 비디오 생성 완료 후 결과 확인

## 주의사항

- 모델 다운로드에 시간이 걸릴 수 있습니다.
- 비디오 생성에는 GPU가 권장됩니다.
- 생성된 비디오는 임시로 저장되며, 새로고침 시 삭제됩니다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 