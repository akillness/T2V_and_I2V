import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video
import platform
import sys
import streamlit as st
from PIL import Image
import io
import warnings
import time # 시간 측정을 위해 추가

# torch 경고 무시
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

def get_device():
    """사용 가능한 디바이스와 데이터 타입을 반환합니다."""
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    # MPS 지원 제거 (필요시 주석 해제)
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     torch_dtype = torch.float16 # MPS는 float16 지원
    else:
        device = "cpu"
        torch_dtype = torch.float32
    return device, torch_dtype

@st.cache_resource
def load_text_pipeline():
    """텍스트-투-비디오 파이프라인을 로드합니다."""
    device, torch_dtype = get_device()
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch_dtype
        
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    if device == "mps":
        pipe.enable_sequential_cpu_offload()
    return pipe


@st.cache_resource
def load_image_pipeline():
    """이미지-투-비디오 파이프라인을 로드합니다."""
    device, torch_dtype = get_device()
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=torch_dtype
        
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    if device == "mps":
        pipe.enable_sequential_cpu_offload()
    return pipe

# @st.cache_resource 제거
def text_to_video(pipe): # pipe 인자 추가
    """텍스트에서 비디오를 생성하고 생성 시간을 반환합니다."""
    device, torch_dtype = get_device()
    st.info(f"Using device: {device} with dtype: {torch_dtype}")

    try:
        # 파이프라인 로드 로직 제거
        # with st.spinner("Loading Text-to-Video pipeline..."):
        #     pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch_dtype, cache_dir="./models")
        #     pipe.to(device)
        # st.success("Pipeline loaded.") # 로드는 main에서 처리

        # 프롬프트 설정
        prompt = st.text_area("Enter prompt:", "A serene lake at sunrise, gentle ripples on the water surface, morning mist slowly rising, birds flying across the golden sky", height=100)
        negative_prompt = st.text_area("Enter negative prompt:", "worst quality, inconsistent motion, blurry, jittery, distorted, watermarks", height=70)

        # 비디오 생성 파라미터 (UI에서 조절 가능하게)
        st.sidebar.header("Generation Parameters")
        width = st.sidebar.slider("Width", min_value=256, max_value=1024, value=704, step=32)
        height = st.sidebar.slider("Height", min_value=256, max_value=1024, value=480, step=32)
        num_frames = st.sidebar.slider("Number of Frames", min_value=9, max_value=257, value=65, step=8)
        if num_frames % 8 != 1: # 8*n + 1 형태 유지
             num_frames = ((num_frames - 1) // 8) * 8 + 1
             st.sidebar.info(f"Adjusted frames to {num_frames} (must be 8*n + 1)")
        num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=50)
        guidance_scale = st.sidebar.slider("Guidance Scale (CFG)", min_value=1.0, max_value=15.0, value=5.0, step=0.5)


        # 메모리 최적화 (파이프라인 로드 시 이미 적용됨)
        # pipe.enable_attention_slicing()
        # if device == "mps":
        #     pipe.enable_sequential_cpu_offload()

        generation_time = 0
        video_frames = None
        output_path = "ltx_text2video.mp4"

        if st.button("Generate Video"):
            with st.spinner(f"Generating video ({num_frames} frames)..."):
                # 비디오 생성 시간 측정 시작
                start_time = time.time()

                # 비디오 생성
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

                # 비디오 생성 시간 측정 종료
                end_time = time.time()
                generation_time = end_time - start_time

                # 비디오 저장
                export_to_video(video_frames, output_path, fps=24)
                st.success(f"Video generated and saved to {output_path}")

                # UI에 비디오 표시
                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()

        return generation_time # 생성 시간 반환

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return 0 # 오류 시 0 반환

# @st.cache_resource 제거
def image_to_video(pipe): # pipe 인자 추가
    """이미지에서 비디오를 생성하고 생성 시간을 반환합니다."""
    device, torch_dtype = get_device()
    st.info(f"Using device: {device} with dtype: {torch_dtype}")

    try:
        # 파이프라인 로드 로직 제거
        # with st.spinner("Loading Image-to-Video pipeline..."):
        #     pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch_dtype, cache_dir="./models")
        #     pipe.to(device)
        # st.success("Pipeline loaded.") # 로드는 main에서 처리

        # 입력 이미지 업로드
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        image = None
        if uploaded_file is not None:
            # 이미지 로드 및 표시
            image_bytes = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption='Uploaded Image.', use_column_width=True)
        else:
            st.warning("Please upload an image file.")
            # 이미지가 없어도 위젯은 표시되어야 하므로 return 0 제거
            # return 0 # 이미지가 없으면 종료


        # 프롬프트 설정
        prompt = st.text_area("Enter prompt:", "The scene transforms...", height=100)
        negative_prompt = st.text_area("Enter negative prompt:", "worst quality, inconsistent motion, blurry, jittery, distorted", height=70)

        # 비디오 생성 파라미터
        st.sidebar.header("Generation Parameters")
        width = st.sidebar.slider("Width", min_value=256, max_value=1024, value=704, step=32)
        height = st.sidebar.slider("Height", min_value=256, max_value=1024, value=480, step=32)
        num_frames = st.sidebar.slider("Number of Frames", min_value=9, max_value=257, value=65, step=8)
        if num_frames % 8 != 1: # 8*n + 1 형태 유지
             num_frames = ((num_frames - 1) // 8) * 8 + 1
             st.sidebar.info(f"Adjusted frames to {num_frames} (must be 8*n + 1)")
        num_inference_steps = st.sidebar.slider("Inference Steps", min_value=10, max_value=100, value=50)
        guidance_scale = st.sidebar.slider("Guidance Scale (CFG)", min_value=1.0, max_value=15.0, value=4.0, step=0.5)
        image_guidance_scale = st.sidebar.slider("Image Guidance Scale", min_value=0.1, max_value=3.0, value=1.5, step=0.1)


        # 메모리 최적화 (파이프라인 로드 시 이미 적용됨)
        # pipe.enable_attention_slicing()
        # if device == "mps":
        #     pipe.enable_sequential_cpu_offload()

        generation_time = 0
        video_frames = None
        output_path = "ltx_img2video.mp4"

        if st.button("Generate Video from Image"):
             if image is None:
                 st.error("Cannot generate video without an uploaded image.")
                 return 0 # 버튼 클릭 시 이미지가 없으면 에러 후 종료

             with st.spinner(f"Generating video ({num_frames} frames) from image..."):
                # 비디오 생성 시간 측정 시작
                start_time = time.time()

                # 비디오 생성
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

                # 비디오 생성 시간 측정 종료
                end_time = time.time()
                generation_time = end_time - start_time

                # 비디오 저장
                export_to_video(video_frames, output_path, fps=24)
                st.success(f"Video generated and saved to {output_path}")

                # UI에 비디오 표시
                video_file = open(output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()

        return generation_time # 생성 시간 반환

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return 0 # 오류 시 0 반환

def main():
    st.set_page_config(layout="wide", page_title="LTX-Video Generator")
    st.title("LTX-Video Generator Demo")

    generation_mode = st.sidebar.selectbox("Select Generation Mode", ["Text-to-Video", "Image-to-Video"])

    total_time = 0
    pipe = None # 파이프라인 변수 초기화

    if generation_mode == "Text-to-Video":
        st.header("Text-to-Video Generation")
        with st.spinner("Loading Text-to-Video pipeline... This may take a while."):
             pipe = load_text_pipeline() # 캐시된 파이프라인 로드
        st.success("Text-to-Video Pipeline loaded.")
        total_time = text_to_video(pipe) # 로드된 파이프라인 전달
    elif generation_mode == "Image-to-Video":
        st.header("Image-to-Video Generation")
        with st.spinner("Loading Image-to-Video pipeline... This may take a while."):
             pipe = load_image_pipeline() # 캐시된 파이프라인 로드
        st.success("Image-to-Video Pipeline loaded.")
        total_time = image_to_video(pipe) # 로드된 파이프라인 전달

    if total_time > 0: # 생성 시간이 0보다 클 때만 표시
        st.info(f"Total video generation time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() # Streamlit 앱 실행 