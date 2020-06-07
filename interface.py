from utils import load_model, decode
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor

model = load_model('cpu')
model.eval()


def recognize(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (200, 64))
    lower = np.array([0, 0, 0])
    upper = np.array([100, 100, 100])
    img = cv2.inRange(img, lower, upper)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = 255 - cv2.dilate(img, element, iterations=2)
    img = to_tensor(img).unsqueeze(0)
    pred = model(img).squeeze().argmax(-1)
    pred = decode(pred)
    return pred


if __name__ == '__main__':
    import sys

    print(recognize(sys.argv[1]))
