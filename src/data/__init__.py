"""Data loading and dataset utilities for detection and uniform recognition.

VisDrone helpers in this package support person-detection preparation only.
They must not be used to invent uniform class labels.
"""

from src.data.visdrone import (
    PERSON_SOURCE_CATEGORY_IDS,
    VISDRONE_CATEGORY_NAMES,
    YOLO_PERSON_CLASS_ID,
    parse_annotation_line,
    visdrone_to_yolo,
)

__all__ = [
    "PERSON_SOURCE_CATEGORY_IDS",
    "VISDRONE_CATEGORY_NAMES",
    "YOLO_PERSON_CLASS_ID",
    "parse_annotation_line",
    "visdrone_to_yolo",
]
