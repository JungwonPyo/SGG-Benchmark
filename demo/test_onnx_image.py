
import cv2
import argparse
import os
import sys

# Add current directory to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.onnx_model import SGG_ONNX_Model

def main(args):
    config_path = args.config
    onnx_path = args.onnx
    rel_conf = args.rel_conf
    box_conf = args.box_conf
    provider = args.provider
    image_path = args.image

    # this will create and load the ONNX model
    model = SGG_ONNX_Model(
        config_path, 
        onnx_path, 
        provider=provider,
        rel_conf=rel_conf, 
        box_conf=box_conf
    )

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Make prediction
    # Resulting image in BGR
    img, graph = model.predict(frame, visu_type='video')

    save_name = "onnx_result.jpg"
    cv2.imwrite(save_name, img)
    print(f"Inference complete. Result saved to {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image test using ONNX")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--onnx', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--provider', type=str, default='CUDAExecutionProvider')
    parser.add_argument('--rel_conf', type=float, default=0.01)
    parser.add_argument('--box_conf', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
