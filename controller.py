from image_utils import (
    LEFT_ARM_OUT,
    RIGHT_ARM_OUT,
    BOTH_ARMS_UP,
)


def trigger_controls(image_categorization_dict):
    if (
        image_categorization_dict[LEFT_ARM_OUT]
        and not image_categorization_dict[RIGHT_ARM_OUT]
    ):
        return
    elif (
        image_categorization_dict[RIGHT_ARM_OUT]
        and not image_categorization_dict[LEFT_ARM_OUT]
    ):
        return
    elif image_categorization_dict[BOTH_ARMS_UP]:
        return
