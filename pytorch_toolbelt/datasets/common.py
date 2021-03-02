__all__ = [
    "IGNORE_LABEL",
    "INPUT_IMAGE_ID_KEY",
    "INPUT_IMAGE_KEY",
    "INPUT_INDEX_KEY",
    "INPUT_MASK_16_KEY",
    "INPUT_MASK_32_KEY",
    "INPUT_MASK_4_KEY",
    "INPUT_MASK_64_KEY",
    "INPUT_MASK_8_KEY",
    "OUTPUT_EMBEDDINGS_KEY",
    "OUTPUT_LOGITS_KEY",
    "OUTPUT_MASK_16_KEY",
    "OUTPUT_MASK_2_KEY",
    "OUTPUT_MASK_32_KEY",
    "OUTPUT_MASK_4_KEY",
    "OUTPUT_MASK_64_KEY",
    "OUTPUT_MASK_8_KEY",
    "OUTPUT_MASK_KEY",
    "TARGET_CLASS_KEY",
    "TARGET_LABELS_KEY",
    "TARGET_MASK_2_KEY",
    "TARGET_MASK_KEY",
    "TARGET_MASK_WEIGHT_KEY",
    "UNLABELED_SAMPLE",
    "name_for_stride",
    "read_image_rgb",
]

# Smaller masks for deep supervision
def name_for_stride(name, stride: int):
    return f"{name}_{stride}"


INPUT_INDEX_KEY = "index"
INPUT_IMAGE_KEY = "image"
INPUT_IMAGE_ID_KEY = "image_id"

TARGET_MASK_KEY = "true_mask"
TARGET_MASK_WEIGHT_KEY = "true_weights"
TARGET_CLASS_KEY = "true_class"
TARGET_LABELS_KEY = "true_labels"


TARGET_MASK_2_KEY = name_for_stride(TARGET_MASK_KEY, 2)
INPUT_MASK_4_KEY = name_for_stride(TARGET_MASK_KEY, 4)
INPUT_MASK_8_KEY = name_for_stride(TARGET_MASK_KEY, 8)
INPUT_MASK_16_KEY = name_for_stride(TARGET_MASK_KEY, 16)
INPUT_MASK_32_KEY = name_for_stride(TARGET_MASK_KEY, 32)
INPUT_MASK_64_KEY = name_for_stride(TARGET_MASK_KEY, 64)

OUTPUT_MASK_KEY = "pred_mask"
OUTPUT_MASK_2_KEY = name_for_stride(OUTPUT_MASK_KEY, 2)
OUTPUT_MASK_4_KEY = name_for_stride(OUTPUT_MASK_KEY, 4)
OUTPUT_MASK_8_KEY = name_for_stride(OUTPUT_MASK_KEY, 8)
OUTPUT_MASK_16_KEY = name_for_stride(OUTPUT_MASK_KEY, 16)
OUTPUT_MASK_32_KEY = name_for_stride(OUTPUT_MASK_KEY, 32)
OUTPUT_MASK_64_KEY = name_for_stride(OUTPUT_MASK_KEY, 64)

OUTPUT_LOGITS_KEY = "pred_logits"
OUTPUT_EMBEDDINGS_KEY = "pred_embeddings"

UNLABELED_SAMPLE = 127
IGNORE_LABEL = 255


def read_image_rgb(fname: str):
    image = cv2.imread(fname)[..., ::-1]
    if image is None:
        raise IOError("Cannot read " + fname)
    return image
