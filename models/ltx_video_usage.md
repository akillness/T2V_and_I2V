# LTX-Video 모델 사용법

LTX-Video는 Lightricks에서 개발한 DiT 기반 비디오 생성 모델입니다. ComfyUI 또는 Hugging Face `diffusers` 라이브러리를 통해 사용할 수 있습니다.

## 주요 사용 방법

### 1. ComfyUI 사용

-   **설치:** ComfyUI Manager를 통해 `ComfyUI-LTXVideo` 커스텀 노드를 설치합니다.
-   **모델 다운로드:** LTX Video 모델, PixArt Text Encoder, T5 Text Encoder를 다운로드하여 `models/checkpoints`, `models/text_encoders/PixArt-XL-2-1024-MS/text_encoder`, `models/text_encoders` 경로에 각각 배치합니다. (자세한 링크는 [ComfyUI Wiki](https://comfyui-wiki.com/en/tutorial/advanced/ltx-video-workflow-step-by-step-guide) 참고)
-   **워크플로우 사용:** 제공되는 Text-to-Video, Image-to-Video, Video-to-Video 워크플로우 파일을 로드하여 사용합니다.
-   **노드 설정:** `LTXVLoader`, `LTXVCLIPModelLoader`, `LTXVModelConfigurator` 등의 노드를 통해 모델, 해상도, 프레임 수, FPS 등을 설정하고 프롬프트를 입력합니다.

### 2. Hugging Face `diffusers` 라이브러리 사용

-   **라이브러리 설치/업데이트:**
    ```bash
    pip install -U git+https://github.com/huggingface/diffusers
    pip install torch # 또는 torch+cuda 등 환경에 맞게 설치
    pip install accelerate # 필요시 설치
    ```
-   **Python 코드 예시 (Text-to-Video):**

    ```python
    import torch
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    # GPU 사용 가능 여부 확인 및 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32 # bfloat16은 CUDA에서 더 효율적

    # 파이프라인 로드
    try:
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch_dtype)
        pipe.to(device)
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print("CPU 또는 다른 dtype으로 재시도합니다.")
        pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
        pipe.to("cpu") # CPU로 명시적 지정

    # 프롬프트 설정
    prompt = "A serene lake at sunrise, gentle ripples on the water surface, morning mist slowly rising, birds flying across the golden sky"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, watermarks"

    # 비디오 생성 파라미터
    width = 704 # 32의 배수
    height = 480 # 32의 배수
    num_frames = 65 # 8*n + 1 형태 (예: 8*8 + 1)
    num_inference_steps = 50
    guidance_scale = 5 # CFG 값 (ComfyUI 예시 참고)

    # 비디오 생성
    # 메모리 부족 오류 방지를 위해 torch.no_grad() 사용 및 필요시 attention slicing 활성화
    pipe.enable_attention_slicing() # 메모리 사용량 감소
    with torch.no_grad():
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).frames[0]

    # 비디오 파일로 저장
    output_video_path = "ltx_generated_video.mp4"
    export_to_video(video_frames, output_video_path, fps=24) # LTX 기본 FPS는 24

    print(f"비디오 생성 완료: {output_video_path}")
    ```

-   **Python 코드 예시 (Image-to-Video):**

    ```python
    import torch
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image
    from PIL import Image
    import requests
    from io import BytesIO

    # GPU 사용 가능 여부 확인 및 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 파이프라인 로드
    try:
        pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch_dtype)
        pipe.to(device)
    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        print("CPU 또는 다른 dtype으로 재시도합니다.")
        pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
        pipe.to("cpu")

    # 입력 이미지 로드 (예시 URL)
    image_url = "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
    try:
        response = requests.get(image_url)
        response.raise_for_status() # 오류 체크
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 오류: {e}")
        # 대체 이미지 또는 로컬 경로 사용 로직 추가 가능
        image = Image.new('RGB', (512, 512), color = 'red') # 예시: 빨간 이미지 생성
    except Exception as e:
        print(f"이미지 처리 오류: {e}")
        image = Image.new('RGB', (512, 512), color = 'blue') # 예시: 파란 이미지 생성


    # 프롬프트 설정
    prompt = "The girl's expression changes to a subtle smile as the fire behind her seems to calm down slightly."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    # 비디오 생성 파라미터
    width = 704
    height = 480
    num_frames = 65
    num_inference_steps = 50
    guidance_scale = 4 # Image-to-Video는 CFG를 약간 낮추는 경향
    image_guidance_scale = 1.5 # 이미지 영향력 조절 (기본값)

    # 비디오 생성
    pipe.enable_attention_slicing()
    with torch.no_grad():
        video_frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
        ).frames[0]

    # 비디오 파일로 저장
    output_video_path = "ltx_img2vid_generated_video.mp4"
    export_to_video(video_frames, output_video_path, fps=24)

    print(f"이미지-비디오 생성 완료: {output_video_path}")

    ```

## 참고 사항

-   **해상도:** 32의 배수 (예: 704x480)
-   **프레임 수:** 8의 배수 + 1 (예: 65, 161, 257)
-   **권장 설정:** 해상도 720x1280 이하, 프레임 수 257 이하
-   **프롬프트:** 영어로 상세하게 작성 (장면, 행동, 세부 묘사 포함)
-   **리소스:**
    -   [Hugging Face 모델 카드](https://huggingface.co/Lightricks/LTX-Video)
    -   [ComfyUI-LTXVideo GitHub](https://github.com/Lightricks/ComfyUI-LTXVideo)
    -   [ComfyUI Wiki 가이드](https://comfyui-wiki.com/en/tutorial/advanced/ltx-video-workflow-step-by-step-guide)
    -   [Diffusers 문서](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx)

## 주의

-   GPU 메모리가 부족할 경우 `torch.cuda.OutOfMemoryError`가 발생할 수 있습니다. 해상도나 프레임 수를 줄이거나, `pipe.enable_attention_slicing()` 또는 `pipe.enable_sequential_cpu_offload()` (더 많은 메모리 절약, 속도 저하)를 사용해 보세요.
-   모델 및 인코더 파일 크기가 크므로 다운로드에 시간이 걸릴 수 있습니다. 