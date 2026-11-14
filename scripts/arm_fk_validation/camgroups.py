"""Find groups of episodes with a static camera by comparing first-frame backgrounds."""
import glob, os, sys
import cv2
import numpy as np

F = "/home/intel/dev/captures/first_frames/431a71f7-50c6-4e38-9e38-d85984dde78b"
cam = sys.argv[1]
ds = sorted(glob.glob(F + "/ep*"))
eps = [int(os.path.basename(d)[2:5]) for d in ds]
g = [cv2.GaussianBlur(cv2.cvtColor(cv2.imread(d + f"/{cam}.png"), cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
     for d in ds]
# edge maps
e = [cv2.Sobel(x, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(x, cv2.CV_32F, 0, 1) ** 2 for x in g]
e = [np.sqrt(x) for x in e]
n = len(g)
for i in range(1, n):
    # phase correlation shift between consecutive frames
    (dx, dy), resp = cv2.phaseCorrelate(e[i - 1], e[i])
    c = np.corrcoef(e[i - 1].ravel(), e[i].ravel())[0, 1]
    print(eps[i - 1], "->", eps[i], "shift %.1f %.1f resp %.2f corr %.2f" % (dx, dy, resp, c))
