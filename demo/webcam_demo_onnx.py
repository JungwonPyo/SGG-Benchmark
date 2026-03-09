import cv2
import argparse
import os
import sys

# Add current directory to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.onnx_model import SGG_ONNX_Model
from sgg_benchmark.utils.miscellaneous import get_path

# main
def main(args):
    config_path = args.config
    onnx_path = args.onnx
    tracking = args.tracking
    rel_conf = args.rel_conf
    box_conf = args.box_conf
    dcs = args.dcs
    save_path = args.save_path
    visu_type = args.visu_type
    provider = args.provider

    # this will create and load the ONNX model
    model = SGG_ONNX_Model(
        config_path, 
        onnx_path, 
        provider=provider,
        dcs=dcs, 
        tracking=tracking, 
        rel_conf=rel_conf, 
        box_conf=box_conf
    )

    # Open the webcam
    cap = cv2.VideoCapture(args.webcam)

    if not cap.isOpened():
        print(f"Error: Could not open webcam {args.webcam}")
        return

    if save_path is not None:
        save_path = os.path.join(get_path(), save_path)
        video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 30, video_size)

    print("Starting webcam demo. Press 'q' to quit, 'p' to pause.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Make prediction
        img, graph = model.predict(frame, visu_type=visu_type)

        # model.predict returns images in BGR for visualization (OpenCV native)
        # So we can display them directly.
        
        if visu_type == 'image' and graph is not None:
            cv2.imshow('Graph', graph)

        # Display the resulting frames
        cv2.imshow('SGG ONNX Demo', img)

        if save_path is not None:
            video_out.write(img)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            cv2.waitKey(-1)
        elif key == ord('q'):
            break

    if save_path is not None:
        video_out.release()

    # When everything is done, release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webcam demo using ONNX")

    parser.add_argument('--config', type=str, required=True, help='Path to the config file (needed for labels)')
    parser.add_argument('--onnx', type=str, required=True, help='Path to the exported ONNX model')
    parser.add_argument('--provider', type=str, default='CUDAExecutionProvider', choices=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'], help='ONNX Runtime provider')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam index')
    parser.add_argument('--tracking', action="store_true", help='Object tracking or not')
    parser.add_argument('--rel_conf', type=float, default=0.01, help='Relation triplet-score threshold (geometric mean of rel×subj×obj)')
    parser.add_argument('--box_conf', type=float, default=0.4, help='Box confidence threshold (non-relation boxes only)')
    parser.add_argument('--dcs', type=int, default=100, help='Dynamic Candidate Selection')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the video')
    parser.add_argument('--visu_type', type=str, default='video', help='Visualization type: video or image')

    args = parser.parse_args()

    # change all relative paths to absolute
    if not os.path.isabs(args.config):
        args.config = os.path.join(get_path(), args.config)
    if not os.path.isabs(args.onnx):
        args.onnx = os.path.join(get_path(), args.onnx)

    main(args)
