import cv2
import numpy as np

from python_eulerian_video_magnification.metadata import MetaData


class Magnify:
    def __init__(self, data: MetaData):
        self._data = data
        self.fps = None

    def load_video(self):
        cap = cv2.VideoCapture(self._in_file_name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        # video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
        # x = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                # video_tensor[x] = frame
                yield frame
                # x += 1
            else:
                break
        # return video_tensor, fps

    def save_video(self, frame_itr) -> None:
        # four_cc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        frame0 = frame_itr.__next__()
        if frame0 is not None:
            [height, width] = frame0.shape[0:2]
            writer = cv2.VideoWriter(self._out_file_name, four_cc, 30, (width, height), 1)
            writer.write(cv2.convertScaleAbs(frame0))
            for frame in frame_itr:
                writer.write(cv2.convertScaleAbs(frame))
            # [height, width] = video_tensor[0].shape[0:2]
            # writer = cv2.VideoWriter(self._out_file_name, four_cc, 30, (width, height), 1)
            # for i in range(0, video_tensor.shape[0]):
            writer.release()

    def do_magnify(self) -> None:
        frames = self.load_video()
        video_tensor = self._magnify_impl(frames, self.fps)
        self.save_video(video_tensor)
        self._data.save_meta_data()

    def _magnify_impl(self, tensor: np.ndarray, fps: int) -> np.ndarray:
        """for generator style, fps use the class property instead of parameter"""
        raise NotImplementedError("This should be overwritten!")

    @property
    def _low(self) -> float:
        return self._data['low']

    @property
    def _high(self) -> float:
        return self._data['high']

    @property
    def _levels(self) -> int:
        return self._data['levels']

    @property
    def _amplification(self) -> float:
        return self._data['amplification']

    @property
    def _in_file_name(self) -> str:
        return self._data['file']

    @property
    def _out_file_name(self) -> str:
        return self._data['target']

