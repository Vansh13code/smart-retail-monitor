import cv2


class VideoReader:

    def __init__(self, path):

        self.cap = cv2.VideoCapture(path)

    def read(self):

        return self.cap.read()

    def release(self):

        self.cap.release()


class VideoWriter:

    def __init__(self, path, fps, width, height):

        self.writer = cv2.VideoWriter(

            path,

            cv2.VideoWriter_fourcc(*"mp4v"),

            fps,

            (width, height)

        )

    def write(self, frame):

        self.writer.write(frame)

    def release(self):

        self.writer.release()