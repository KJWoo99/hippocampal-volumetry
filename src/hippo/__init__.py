"""Hippocampal 3D segmentation & clinical volumetry pipeline."""

__version__ = "0.1.0"

# 클래스 정의: 0=배경, 1=해마 anterior, 2=해마 posterior
LABELS = {0: "background", 1: "anterior", 2: "posterior"}
NUM_CLASSES = 3
FOREGROUND_LABELS = (1, 2)
