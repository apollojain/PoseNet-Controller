import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import time
import argparse

import posenet

# left arm
LEFT_SHOULDER = "leftShoulder"
LEFT_ELBOW = "leftElbow"
LEFT_WRIST = "leftWrist"
# right arm
RIGHT_SHOULDER = "rightShoulder"
RIGHT_ELBOW = "rightElbow"
RIGHT_WRIST = "rightWrist"
# slope values
STRAIGHT_SLOPE_THRESH = 0.5
VERTICAL_SLOPE_THRESH = 2


def process_frame(
    sess,
    model_outputs,
    output_stride,
    input_image,
    display_image,
    output_scale,
    max_pose_detections,
    min_pose_score,
    min_part_score,
):
    (
        heatmaps_result,
        offsets_result,
        displacement_fwd_result,
        displacement_bwd_result,
    ) = sess.run(model_outputs, feed_dict={"image:0": input_image})

    (
        pose_scores,
        keypoint_scores,
        keypoint_coords,
    ) = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=max_pose_detections,
        min_pose_score=min_pose_score,
    )

    keypoint_coords *= output_scale

    # TODO this isn't particularly fast, use GL for drawing and display someday...
    overlay_image = posenet.draw_skel_and_kp(
        display_image,
        pose_scores,
        keypoint_scores,
        keypoint_coords,
        min_pose_score=min_pose_score,
        min_part_score=min_part_score,
    )

    parts_dict = {}
    for pi in range(len(pose_scores)):
        if pose_scores[pi] == 0.0:
            break
        parts_dict = {
            posenet.PART_NAMES[ki]: (s, c)
            for ki, (s, c) in enumerate(
                zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])
            )
            if s > min_part_score
        }
    return overlay_image, parts_dict


def abs_slope(parts_dict, part0, part1):
    _, coords0 = parts_dict[part0]
    _, coords1 = parts_dict[part1]
    x0, y0 = coords0
    x1, y1 = coords1
    return np.abs(float(y1 - y0) / float(x1 - x0))


def left_arm_parts_exist(parts_dict):
    return (
        LEFT_WRIST in parts_dict
        and LEFT_ELBOW in parts_dict
        and LEFT_SHOULDER in parts_dict
    )


def right_arm_parts_exist(parts_dict):
    return (
        RIGHT_WRIST in parts_dict
        and RIGHT_ELBOW in parts_dict
        and RIGHT_SHOULDER in parts_dict
    )


def part_left_of(parts_dict, part_left, part_right):
    _, coords0 = parts_dict[part_left]
    _, coords1 = parts_dict[part_right]
    x0, _ = coords0
    x1, _ = coords1
    return x0 < x1


def part_right_of(parts_dict, part_left, part_right):
    return not part_left(parts_dict, part_left, part_right)


def part_above(parts_dict, part_above, part_below):
    _, coords0 = parts_dict[part_left]
    _, coords1 = parts_dict[part_right]
    _, y0 = coords0
    _, y1 = coords1
    return y0 < y1


def left_arm_extended(parts_dict):
    # check that all relevant body parts are in the dict
    if not left_arm_parts_exist(parts_dict):
        return False
    # make sure wrist is left of elbow and elbow is left of shoulder
    if not (
        part_left_of(parts_dict, LEFT_WRIST, LEFT_ELBOW)
        and part_left_of(parts_dict, LEFT_ELBOW, LEFT_SHOULDER)
    ):
        return False
    # slope(leftElbow, leftShoulder) close to 0 and slope(leftElbow, leftWrist) close to 0
    return (
        abs_slope(parts_dict, LEFT_ELBOW, LEFT_SHOULDER) < STRAIGHT_SLOPE_THRESH
        and abs_slope(parts_dict, LEFT_SHOULDER, LEFT_WRIST) < STRAIGHT_SLOPE_THRESH
    )


def right_arm_extended(parts_dict):
    # check that all relevant body parts are in the dict
    if not right_arm_parts_exist(parts_dict):
        return False
    # make sure wrist is right of elbow and elbow is right of shoulder
    if not (
        part_right_of(parts_dict, RIGHT_WRIST, RIGHT_ELBOW)
        and part_right_of(parts_dict, RIGHT_ELBOW, RIGHT_SHOULDER)
    ):
        return False
    # slope(rightElbow, rightShoulder) close to 0 and slope(rightElbow, rightWrist) close to 0
    return (
        abs_slope(parts_dict, RIGHT_ELBOW, RIGHT_SHOULDER) < STRAIGHT_SLOPE_THRESH
        and abs_slope(parts_dict, RIGHT_SHOULDER, RIGHT_WRIST) < STRAIGHT_SLOPE_THRESH
    )


def left_arm_up(parts_dict):
    # Relevant body parts: ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist")
    if not left_arm_parts_exist(parts_dict):
        return False
    # leftWrist above leftElbow and leftElbow above leftShoulder
    if not (
        part_above(parts_dict, LEFT_WRIST, LEFT_ELBOW)
        and part_above(parts_dict, LEFT_ELBOW, LEFT_SHOULDER)
    ):
        return False
    # slope(leftElbow, leftShoulder) is close to infinity and slope(leftElbow, leftWrist) is close to infinity
    return (
        abs_slope(parts_dict, LEFT_ELBOW, LEFT_SHOULDER) > VERTICAL_SLOPE_THRESH
        and abs_slope(parts_dict, LEFT_SHOULDER, LEFT_WRIST) > VERTICAL_SLOPE_THRESH
    )


def right_arm_up(parts_dict):
    # Relevant body parts: ("rightShoulder", "rightElbow"), ("leftElbow", "rightWrist")
    if not right_arm_parts_exist(parts_dict):
        return False
    # rightWrist above rightElbow and rightElbow above rightShoulder
    if not (
        part_above(parts_dict, RIGHT_WRIST, RIGHT_ELBOW)
        and part_above(parts_dict, RIGHT_ELBOW, RIGHT_SHOULDER)
    ):
        return False
    # slope(rightElbow, rightShoulder) is close to infinity and slope(rightElbow, rightWrist) is close to infinity
    return (
        abs_slope(parts_dict, RIGHT_ELBOW, RIGHT_SHOULDER) > VERTICAL_SLOPE_THRESH
        and abs_slope(parts_dict, RIGHT_SHOULDER, RIGHT_WRIST) > VERTICAL_SLOPE_THRESH
    )


def both_arms_up(parts_dict):
    return right_arm_up(parts_dict) and left_arm_up(parts_dict)


def get_image_categorization_dict(parts_dict):
    return {
        LEFT_ARM_OUT: left_arm_extended(parts_dict),
        RIGHT_ARM_OUT: right_arm_extended(parts_dict),
        BOTH_ARMS_UP: both_arms_up(parts_dict),
    }