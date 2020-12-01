import cv2
import numpy as np

class PossivelPlaca:
    def __init__(self):
        self.imgPlaca = None
        self.imgEscalaDeCinza = None
        self.imgThreshold = None

        self.rrLocationOfPlacaInScene = None

        self.strCaracteres = ""