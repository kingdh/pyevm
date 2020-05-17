import cv2
import numpy as np
import collections

from python_eulerian_video_magnification.filter import temporal_ideal_filter
from python_eulerian_video_magnification.magnify import Magnify
from python_eulerian_video_magnification.pyramid import gaussian_video


class MagnifyColor(Magnify):
    def _magnify_impl(self, frames, fps: int):
        gau_video = gaussian_video(frames, levels=self._levels)
        window_size = self._data['window_size']
        window = collections.deque([], maxlen=2*window_size)

        step = window_size
        i = step
        j=0
        for frame in gau_video:
            window.append(frame)
            if len(window)==window.maxlen:
                if i >= step:
                    tensor = np.array(window, dtype='float')
                    filtered_tensor = temporal_ideal_filter(tensor[0:window_size], self._low, self._high, self.fps)
                    amplified_video = self._amplify_video(filtered_tensor)
                    assert(filtered_tensor.shape[0] == window_size)
                    assert(amplified_video.shape[0]==filtered_tensor.shape[0])
                    j += amplified_video.shape[0]
                    sliced = self._reconstruct_video(amplified_video, tensor[0:window_size])
                    assert(sliced.shape[0] == window_size)
                    print("processed ", j, " frames")
                    for f in sliced:
                        yield f
                    i = 0
                else:
                    i += 1
        if i>0:
            tensor = np.array(window, dtype='float')
            filtered_tensor = temporal_ideal_filter(tensor[-(window_size+i):-1], self._low, self._high, self.fps)
            amplified_video = self._amplify_video(filtered_tensor)
            sliced = self._reconstruct_video(amplified_video, tensor[-(window_size+i):-1])
            j += window_size + i
            print("processed ", j, " frames")
            for f in sliced:
                yield f

    def _amplify_video(self, gaussian_vid):
        return gaussian_vid * self._amplification

    def _reconstruct_video(self, amp_video, origin_video):
        origin_video_shape = origin_video.shape[1:]
        for i in range(0, amp_video.shape[0]):
            img = amp_video[i]
            for x in range(self._levels):
                img = cv2.pyrUp(img)  # this doubles the dimensions of img each time
            # ensure that dimensions are equal
            origin_video[i] += self._correct_dimensionality_problem_after_pyr_up(img, origin_video_shape)
        return origin_video

    def _correct_dimensionality_problem_after_pyr_up(self, img: np.ndarray, origin_video_frame_shape) -> np.ndarray:
        if img.shape != origin_video_frame_shape:
            return np.resize(img, origin_video_frame_shape)
        else:
            return img

    # no use now
    def principal_component_analysis(self, tensor: np.ndarray):
        # Data matrix tensor, assumes 0-centered
        n, m = tensor.shape
        assert np.allclose(tensor.mean(axis=0), np.zeros(m))
        # Compute covariance matrix
        covariance_matrix = np.dot(tensor.T, tensor) / (n - 1)
        # Eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(covariance_matrix)
        # Project tensor onto PC space
        X_pca = np.dot(tensor, eigen_vecs)
        return X_pca
