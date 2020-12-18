import keyboard
from image_utils import (
    LEFT_ARM_OUT,
    RIGHT_ARM_OUT,
    BOTH_ARMS_UP,
    LEFT_ARM_UP,
    RIGHT_ARM_UP,
)

ONE_ARM_UP = "one_arm_up"

SPACE = "space, space, space, space"
LEFT = "left, left, left, left, left, left, left, left"
RIGHT = "right, right, right, right, right, right, right, right"


def trigger_controls(image_categorization_dict):
    if (
        image_categorization_dict[LEFT_ARM_OUT]
        and not image_categorization_dict[RIGHT_ARM_OUT]
    ):
        print(LEFT_ARM_OUT)
        keyboard.press_and_release(LEFT)
    elif (
        image_categorization_dict[RIGHT_ARM_OUT]
        and not image_categorization_dict[LEFT_ARM_OUT]
    ):
        print(RIGHT_ARM_OUT)
        keyboard.press_and_release(RIGHT)
    elif (
        image_categorization_dict[LEFT_ARM_UP]
        or image_categorization_dict[RIGHT_ARM_UP]
    ):
        print(BOTH_ARMS_UP)
        keyboard.press_and_release(SPACE)
