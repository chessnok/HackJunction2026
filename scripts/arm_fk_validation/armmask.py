"""Arm masks: floor camera = purple hue; top camera = dark pixels on the paper (arm is backlit)."""
import cv2
import numpy as np


def mask_floor(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    m = (h >= 118) & (h <= 165) & (s >= 80) & (v >= 30)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def mask_top(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    green = (h >= 35) & (h <= 95) & (s >= 90)
    blueish = (h >= 95) & (h <= 112) & (s >= 120)  # bricks (blue), not the arm
    m = (v <= 75) & ~green & ~blueish
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def mask_top2(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    purple = (h >= 115) & (h <= 165) & (s >= 60) & (v >= 30)
    green = (h >= 35) & (h <= 95) & (s >= 90)
    blueish = (h >= 92) & (h <= 114) & (s >= 100)
    dark = (v <= 50) & ~green & ~blueish
    m = purple | dark
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


MASKS = {"top": mask_top2, "floor": mask_floor}
