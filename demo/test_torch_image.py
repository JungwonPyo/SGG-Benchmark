import cv2
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.demo_model import SGG_Model

def main(args):
    model = SGG_Model(
        args.config,
        args.weights,
        rel_conf=args.rel_conf,
        box_conf=args.box_conf,
    )

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not read image {args.image}")
        return

    img, graph = model.predict(frame, visu_type='video')

    save_name = "torch_result.jpg"
    cv2.imwrite(save_name, img)
    print(f"Inference complete. Result saved to {save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static image test using PyTorch model")
    parser.add_argument('--config',  type=str, required=True, help='Path to config.yml')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--image',   type=str, required=True, help='Path to input image')
    parser.add_argument('--rel_conf', type=float, default=0.01)
    parser.add_argument('--box_conf', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
