import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import numpy as np
import time
import argparse

import posenet
import image_utils

import unittest
from config import (
    MAX_POSE_DETECTIONS,
    MIN_POSE_SCORE,
    MIN_PART_SCORE,
)


class TestPoseNet(unittest.TestCase):
    def get_parts_dict(self, image_filepath):
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg["output_stride"]
            scale_factor = 0.7125
            input_image, display_image, output_scale = posenet.read_imgfile(
                image_filepath, scale_factor=scale_factor, output_stride=output_stride
            )
            _, parts_dict = image_utils.process_frame(
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

            return parts_dict

    def test_right_arm_out(self):
        right_arm_out_file = "./images/right_arm_out.jpg"
        parts_dict = self.get_parts_dict(right_arm_out_file)
        image_categorization_dict = image_utils.get_image_categorization_dict(
            parts_dict
        )
        self.assertEqual(image_categorization_dict[image_utils.LEFT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_ARM_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_ARMS_UP], False)

    def test_left_arm_out(self):
        left_arm_out_file = "./images/left_arm_out.jpg"
        parts_dict = self.get_parts_dict(left_arm_out_file)
        image_categorization_dict = image_utils.get_image_categorization_dict(
            parts_dict
        )
        self.assertEqual(image_categorization_dict[image_utils.LEFT_ARM_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_ARMS_UP], False)

    def test_both_arms_up(self):
        both_arms_up_file = "./images/both_arms_up.jpg"
        parts_dict = self.get_parts_dict(both_arms_up_file)
        image_categorization_dict = image_utils.get_image_categorization_dict(
            parts_dict
        )
        self.assertEqual(image_categorization_dict[image_utils.LEFT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_ARMS_UP], True)

    def test_both_arms_down(self):
        both_arms_down_file = "./images/both_arms_down.jpg"
        parts_dict = self.get_parts_dict(both_arms_down_file)
        image_categorization_dict = image_utils.get_image_categorization_dict(
            parts_dict
        )
        self.assertEqual(image_categorization_dict[image_utils.LEFT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_ARM_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_ARMS_UP], False)


if __name__ == "__main__":
    unittest.main()
