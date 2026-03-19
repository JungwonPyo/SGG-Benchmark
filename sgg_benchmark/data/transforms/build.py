# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

def build_transforms(cfg, is_train=True):
    padding = cfg.input.padding

    if is_train:
        min_size = cfg.input.img_size[1]  # H
        max_size = cfg.input.img_size[0]  # W
        flip_horizontal_prob = cfg.input.flip_prob_train
        flip_vertical_prob = cfg.input.vertical_flip_prob_train
        brightness = cfg.input.brightness
        contrast = cfg.input.contrast
        saturation = cfg.input.saturation
        hue = cfg.input.hue
    else:
        min_size = cfg.input.img_size[1]  # H
        max_size = cfg.input.img_size[0]  # W
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.input.to_bgr255
    normalize_transform = T.Normalize(
        mean=cfg.input.pixel_mean, std=cfg.input.pixel_std, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    if padding:
        transform = T.Compose(
            [
                color_jitter,
                T.LetterBox(new_shape=(min_size, max_size)),
                T.ToTensorYOLO(),
            ]
        )
    else:
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
