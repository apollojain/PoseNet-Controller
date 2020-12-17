import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import numpy as np
import time
import argparse

import posenet
import image_utils


MAX_POSE_DETECTIONS = 1
MIN_POSE_SCORE = 0.15
MIN_PART_SCORE = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, default=101)
parser.add_argument("--cam_id", type=int, default=0)
parser.add_argument("--cam_width", type=int, default=1280)
parser.add_argument("--cam_height", type=int, default=720)
parser.add_argument("--scale_factor", type=float, default=0.7125)
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Optionally use a video file instead of a live camera",
)
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg["output_stride"]

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride
            )

            print(np.shape(input_image))
            print(np.shape(display_image))

            input_image[0, :, :, :] = cv2.flip(input_image[0, :, :, :], 1)
            display_image = cv2.flip(display_image, 1)

            overlay_image, parts_dict = image_utils.process_frame(
                sess,
                model_outputs,
                output_stride,
                input_image,
                display_image,
                output_scale,
                MAX_POSE_DETECTIONS,
                MIN_POSE_SCORE,
                MIN_PART_SCORE,
            )

            cv2.imshow("posenet", overlay_image)
            print("Parts Dict:", parts_dict)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print("Average FPS: ", frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
