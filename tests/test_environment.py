"""Smoke tests: major libraries import successfully."""

from __future__ import annotations


def test_import_torch():
    import torch

    assert torch.__version__


def test_import_torchvision():
    import torchvision

    assert torchvision.__version__


def test_import_ultralytics():
    import ultralytics

    assert ultralytics is not None


def test_import_cv2():
    import cv2

    assert cv2.__version__


def test_import_numpy():
    import numpy as np

    assert np.__version__


def test_import_pandas():
    import pandas as pd

    assert pd.__version__


def test_import_sklearn():
    import sklearn

    assert sklearn.__version__


def test_import_matplotlib():
    import matplotlib

    assert matplotlib.__version__


def test_import_albumentations():
    import albumentations

    assert albumentations.__version__


def test_import_pil():
    from PIL import Image

    assert Image is not None


def test_import_yaml():
    import yaml

    assert yaml is not None


def test_import_tqdm():
    import tqdm

    assert tqdm.__version__
