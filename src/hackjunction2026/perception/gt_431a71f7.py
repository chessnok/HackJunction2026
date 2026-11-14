"""Hand-annotated ground truth for dataset 431a71f7 (agent A, 2026-11-14).

How it was made: for each evaluation frame (scripts/perception_export_eval_frames.py) the top and
floor images were cropped around the brick, upscaled 2x with a 10 px grid labelled in full-image
pixels, and the brick's visible extent was read off by eye (precision about +-2 px; the bottom
edge, which defines the table position, is the most careful reading). No detector output was shown
while annotating. Colour comes from the image, cross-checked with the episode's task name.

bbox = [x0, y0, x1, y1] in image pixels (visible extent; for partial occlusion by the gripper the
brick outline that can be inferred). occ: 0 = free, 1 = partly hidden by the gripper,
2 = mostly hidden. lifted: the brick is in the gripper above the table (both views move up).
"""

# (episode, frame) -> {cam: (color, bbox, occ)}, plus "lifted"
GT = {
    (4, 0): {"top": ("blue", [480, 141, 558, 209], 0), "floor": ("blue", [485, 78, 547, 124], 0)},
    (4, 256): {"top": ("blue", [480, 142, 558, 210], 0), "floor": ("blue", [485, 78, 547, 124], 0)},
    (4, 418): {"top": ("blue", [511, 104, 583, 199], 1), "floor": ("blue", [466, 68, 513, 127], 1)},
    (8, 0): {"top": ("yellow", [543, 75, 611, 140], 0), "floor": ("yellow", [320, 89, 380, 146], 0)},
    (8, 130): {"top": ("yellow", [543, 76, 611, 140], 0), "floor": ("yellow", [320, 89, 380, 146], 0)},
    (8, 212): {"top": ("yellow", [550, 79, 618, 140], 1), "floor": ("yellow", [319, 89, 378, 146], 1)},
    (13, 0): {"top": ("red", [513, 48, 568, 106], 0), "floor": ("red", [264, 111, 319, 180], 0)},
    (13, 131): {"top": ("red", [514, 47, 568, 106], 0), "floor": ("red", [264, 111, 319, 180], 0)},
    (13, 215): {"top": ("red", [525, 45, 575, 101], 1), "floor": ("red", [244, 118, 312, 183], 1)},
    (19, 0): {"top": ("blue", [480, 157, 560, 221], 0), "floor": ("blue", [442, 85, 500, 118], 0)},
    (19, 123): {"top": ("blue", [480, 157, 560, 220], 0), "floor": ("blue", [442, 86, 500, 118], 0)},
    (19, 202): {"top": ("blue", [519, 135, 592, 197], 1), "floor": ("blue", [430, 63, 487, 109], 1), "lifted": True},
    (22, 0): {"top": ("blue", [442, 140, 510, 202], 0), "floor": ("blue", [454, 94, 513, 127], 0)},
    (22, 173): {"top": ("blue", [442, 140, 510, 202], 0), "floor": ("blue", [454, 94, 513, 127], 0)},
    (22, 284): {"top": ("blue", [489, 133, 552, 200], 1), "floor": ("blue", [421, 86, 487, 125], 1)},
    (25, 0): {"top": ("green", [393, 81, 460, 137], 0), "floor": ("green", [432, 111, 508, 165], 0)},
    (25, 152): {"top": ("green", [393, 81, 460, 136], 0), "floor": ("green", [432, 111, 508, 165], 0)},
    (25, 249): {"top": ("green", [411, 85, 478, 140], 1), "floor": ("green", [421, 111, 501, 159], 1)},
    (29, 0): {"top": ("green", [432, 40, 484, 78], 0), "floor": ("green", [280, 166, 358, 236], 0)},
    (29, 125): {"top": ("green", [432, 40, 485, 78], 0), "floor": ("green", [280, 166, 358, 237], 0)},
    (29, 204): {"top": ("green", [432, 39, 489, 83], 1), "floor": ("green", [280, 164, 356, 235], 1)},
    (33, 0): {"top": ("green", [494, 43, 549, 116], 0), "floor": ("green", [290, 105, 376, 187], 0)},
    (33, 127): {"top": ("green", [494, 44, 550, 116], 0), "floor": ("green", [290, 105, 376, 187], 0)},
    (33, 207): {"top": ("green", [494, 43, 545, 116], 1), "floor": ("green", [291, 104, 377, 188], 1)},
    (36, 0): {"top": ("green", [458, 134, 587, 210], 0), "floor": ("green", [431, 69, 516, 117], 0)},
    (36, 146): {"top": ("green", [458, 134, 587, 211], 0), "floor": ("green", [431, 69, 516, 117], 0)},
    (36, 238): {"top": ("green", [463, 128, 582, 220], 1), "floor": ("green", [430, 64, 516, 115], 1)},
    (41, 0): {"top": ("blue+red", [508, 134, 646, 234], 0), "floor": ("blue+red", [419, 61, 494, 112], 0)},
    (41, 147): {"top": ("blue+red", [508, 134, 646, 233], 0), "floor": ("blue+red", [419, 61, 494, 112], 0)},
    (41, 241): {"top": ("blue+red", [512, 136, 642, 241], 1), "floor": ("blue+red", [419, 58, 493, 108], 1)},
    (44, 0): {"top": ("blue+yellow", [433, 26, 479, 98], 0), "floor": ("blue+yellow", [335, 104, 421, 206], 0)},
    (44, 142): {"top": ("blue+yellow", [433, 26, 479, 98], 0), "floor": ("blue+yellow", [335, 104, 421, 206], 0)},
    (44, 232): {"top": ("blue+yellow", [430, 36, 497, 103], 1), "floor": ("blue+yellow", [346, 98, 414, 193], 1)},
    (46, 0): {"top": ("red+blue", [555, 64, 638, 151], 0), "floor": ("red+blue", [326, 64, 388, 136], 0)},
    (46, 157): {"top": ("red+blue", [555, 64, 638, 151], 0), "floor": ("red+blue", [326, 64, 388, 136], 0)},
    (46, 256): {"top": ("red+blue", [555, 61, 633, 153], 1), "floor": ("red+blue", [328, 64, 388, 138], 1)},
    (49, 0): {"top": ("blue", [620, 42, 690, 122], 0), "floor": ("blue", [232, 64, 291, 149], 0)},
    (49, 173): {"top": ("blue", [620, 42, 690, 122], 0), "floor": ("blue", [232, 64, 291, 149], 0)},
    (49, 284): {"top": ("blue", [616, 40, 693, 124], 2), "floor": ("blue", [229, 66, 282, 148], 1)},
    (53, 0): {"top": ("red", [578, 150, 676, 227], 0), "floor": ("red", [372, 80, 421, 117], 0)},
    (53, 185): {"top": ("red", [578, 150, 676, 227], 0), "floor": ("red", [372, 80, 421, 117], 0)},
    (53, 303): {"top": ("red", [577, 152, 672, 227], 1), "floor": ("red", [371, 80, 418, 117], 1)},
    (58, 0): {"top": ("blue", [654, 135, 740, 224], 0), "floor": ("blue", [339, 68, 378, 115], 0)},
    (58, 133): {"top": ("blue", [654, 135, 740, 224], 0), "floor": ("blue", [339, 68, 378, 115], 0)},
    (58, 217): {"top": ("blue", [668, 150, 728, 224], 2), "floor": ("blue", [339, 75, 377, 115], 1)},
    (63, 0): {"top": ("red", [425, 87, 539, 153], 0), "floor": ("red", [321, 100, 427, 160], 0)},
    (63, 170): {"top": ("red", [425, 86, 539, 154], 0), "floor": ("red", [321, 100, 427, 160], 0)},
    (63, 279): {"top": ("red", [440, 54, 557, 132], 1), "floor": ("red", [311, 71, 415, 132], 1), "lifted": True},
}
