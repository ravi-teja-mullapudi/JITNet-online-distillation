import os
import cv2
import time
from random import randint

class VideoInputStream:
    def __init__(self, stream_path, start_frame = 0,
                 loop = False, reset_frame = 0):
        assert(os.path.isfile(stream_path))
        self.cap = cv2.VideoCapture(stream_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.loop = loop
        self.reset_frame = reset_frame
        self.start_frame = start_frame

        assert(self.start_frame >= 0 and self.start_frame < self.length)
        assert(self.reset_frame >= 0 and self.reset_frame < self.length)

        # Seek to the start frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            if not self.loop:
                raise StopIteration()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.reset_frame)
                ret, frame = self.cap.read()

        return frame

    def __iter__(self):
        return self

    next = __next__
