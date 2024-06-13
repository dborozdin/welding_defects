import cv2
from PIL import Image

from ultralytics import YOLO

model = YOLO("welding.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")
#results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("SampleV1_1_mp4-1_jpg.rf.3f50c974a91c4e6348dd49491f06def8.jpg")
results = model.predict(source=im1, classes=[0], save=True, show=True)  # save plotted images

# from ndarray
#im2 = cv2.imread("Sample2.jpg")
#results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
#results = model.predict(source=[im1, im2])
