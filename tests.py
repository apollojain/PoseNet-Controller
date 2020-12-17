import unittest
from webcam_demo import (
    MAX_POSE_DETECTIONS,
    MIN_POSE_SCORE,
    MIN_PART_SCORE,
)

class TestPoseNet(object):

    
    def get_parts_dict(self, image_filepath):
        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(101, sess)
            output_stride = model_cfg["output_stride"]
            scale_factor = 0.7125
            input_image, draw_image, output_scale = posenet.read_imgfile(
                image_filepath, scale_factor=scale_factor, output_stride=output_stride
            )
            _, parts_dict = image_utils.process_frame((
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
         

    def test_right_hand_out(self):
        right_hand_out_file = 
        parts_dict = self.get_parts_dict(right_hand_out_file)
        image_categorization_dict = get_image_categorization_dict(parts_dict)
        self.assertEqual(image_categorization_dict[image_utils.LEFT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_HAND_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_HANDS_UP], False)

    def test_left_hand_out(self):
        left_hand_out_file = 
        parts_dict = self.get_parts_dict(left_hand_out_file)
        image_categorization_dict = get_image_categorization_dict(parts_dict)
        self.assertEqual(image_categorization_dict[image_utils.LEFT_HAND_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_HANDS_UP], False)

    def test_both_hands_up(self):
        both_hands_up_file = 
        parts_dict = self.get_parts_dict(both_hands_up_file)
        image_categorization_dict = get_image_categorization_dict(parts_dict)
        self.assertEqual(image_categorization_dict[image_utils.LEFT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_HANDS_UP], True)

    def test_both_hands_down(self):
        both_hands_down_file = 
        parts_dict = self.get_parts_dict(both_hands_down_file)
        image_categorization_dict = get_image_categorization_dict(parts_dict)
        self.assertEqual(image_categorization_dict[image_utils.LEFT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_HAND_OUT], False)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_HANDS_UP], False)

    def test_both_hands_out(self):
        both_hands_out_file = 
        parts_dict = self.get_parts_dict(both_hands_out_file)
        image_categorization_dict = get_image_categorization_dict(parts_dict)
        self.assertEqual(image_categorization_dict[image_utils.LEFT_HAND_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.RIGHT_HAND_OUT], True)
        self.assertEqual(image_categorization_dict[image_utils.BOTH_HANDS_UP], False)