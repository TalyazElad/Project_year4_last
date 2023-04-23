import cv2
import os
import streamlit as st
import niqe_runner
from brisque_new import compute_score
import warnings

# Ignore all warnings
warnings.simplefilter("ignore")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(video_name, exist_ok=True)
    frame_count = 0
    niqe_sum = 0
    pique_sum = 0
    brisque_sum = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_name = f"{video_name}_frame{frame_count}.jpg"
            cv2.imwrite(os.path.join(video_name, frame_name), frame)
            frame_count += 1
            niqe_sum += float(niqe_runner.runner(os.path.join(video_name, frame_name)))
            pique_sum += float(niqe_runner.pique_runner(os.path.join(video_name, frame_name)))
            brisque_sum += float(compute_score(os.path.join(video_name, frame_name)))
        else:
            break
    cap.release()
    st.success(f"Frames saved to folder {video_name}")
    niqe_avg = niqe_sum/frame_count
    pique_avg = pique_sum/frame_count
    brisque_avg = brisque_sum/frame_count
    st.success(f"niqe score : {niqe_avg}, pique score : {pique_avg}, brisque score : {brisque_avg}")

def main():
    st.title("Video quality evaluation")

    uploaded_file = st.file_uploader("Select an MP4 video file", type="mp4")

    if uploaded_file is not None:
        video_path = "uploaded.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        extract_frames(video_path)
        os.remove(video_path)

if __name__ == "__main__":
    main()