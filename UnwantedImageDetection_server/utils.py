import hashlib
import numpy as np
import cv2
import base64


def hashkey(key):
    return hashlib.sha256(key.encode()).hexdigest()

def base642array(b64):
    b64 = b64.split(',')[-1]
    image = base64.b64decode(b64)
    image = np.frombuffer(image, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)
