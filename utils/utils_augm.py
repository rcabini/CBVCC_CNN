import tqdm, re
import random
import pathlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32) # convert dtype and normalize [0,1)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    video_reader = cv2.VideoCapture(video_path)
    video = []
    k = 0
    while True:
        ret, frame = video_reader.read()
        if frame is None: 
            if k!=20:
                print('NOO', k)
            break
        else: 
            frame = format_frames(frame, output_size)
            video.append(frame) #RGB
            k+=1
    result = np.array(video)[..., [2, 1, 0]]
    return result

class FrameGenerator:
    def __init__(self, path, n_frames, size, training = False, augmentation = False, undersample_factor=0):
        """ Returns a set of frames with their associated label. 

          Args:
            path: Video file paths.
            n_frames: Number of frames. 
            training: Boolean to determine if training dataset is being created.
        """
        self.path = path
        self.n_frames = n_frames
        self.output_size = (size,size)
        self.training = training
        self.augmentation = augmentation
        self.undersample_factor = undersample_factor
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def _rotate(self, frame, angle):
        return tf.keras.preprocessing.image.random_rotation(frame, angle, row_axis=0, col_axis=1, channel_axis=2)
    
    def _zoom(self, frame, zoom_factor):
        target_height = int(frame.shape[0] / zoom_factor)
        target_width = int(frame.shape[1] / zoom_factor)
        frame_zoomed = tf.image.resize(frame, [target_height, target_width], method=tf.image.ResizeMethod.BICUBIC)
        # Resize the frame back to the original size using nearest neighbor interpolation
        frame_zoomed = tf.image.resize(frame_zoomed, [frame.shape[0], frame.shape[1]], method=tf.image.ResizeMethod.BICUBIC)
        return frame_zoomed
    
    def _shift(self, frame, shift_factor):
        shift_pixels = int(frame.shape[0] * shift_factor)
        frame = tf.roll(frame, shift_pixels, axis=0)
        frame = tf.roll(frame, shift_pixels, axis=1)
        return frame
    
    def _contrast_stretching(self, frame, contrast_factor):
        frame = tf.image.adjust_contrast(frame, contrast_factor)
        return frame
        
    def _horizontal_flip(self, frame):
        return tf.image.flip_left_right(frame)

    def _vertical_flip(self, frame):
        return tf.image.flip_up_down(frame)
    
    def _random_transform(self, frames):
        # Random rotation
        angle = np.random.uniform(-90, 90)
        frames = [self._rotate(frame, angle) for frame in frames]
        # Random zoom
        zoom_factor = np.random.uniform(0.9, 1.1)
        frames = [self._zoom(frame, zoom_factor) for frame in frames]
        # Random shift
        #shift_factor = np.random.uniform(-0.1, 0.1)
        #frames = [self._shift(frame, shift_factor) for frame in frames]
        # Random contrast stretching
        contrast_factor = np.random.uniform(0.5, 1.5)
        frames = [self._contrast_stretching(frame, contrast_factor) for frame in frames]
        # Random horizontal flip
        if np.random.choice([True, False]):
            frames = [self._horizontal_flip(frame) for frame in frames]
        # Random vertical flip
        if np.random.choice([True, False]):
            frames = [self._vertical_flip(frame) for frame in frames]
        return frames

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()

        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)
        class_0_videos = [pair for pair in pairs if pair[1] == '0']
        class_1_videos = [pair for pair in pairs if pair[1] == '1']
        print('++++++', len(class_0_videos))
        print('++++++', len(class_1_videos))
        if self.undersample_factor > 1:
            class_0_videos = random.sample(class_0_videos, len(class_0_videos) // self.undersample_factor)
        print('++++++', len(class_0_videos))
        pairs = class_0_videos + class_1_videos

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames, output_size=self.output_size, frame_step = 1)

            if not np.all(video_frames==0) and not np.isinf(video_frames).any() and not np.isnan(video_frames).any() and not np.any(video_frames<0):
                if self.augmentation == True:
                    video_frames = self._random_transform(video_frames)
                    min_val = tf.reduce_min(video_frames)
                    max_val = tf.reduce_max(video_frames)
                    video_frames = (video_frames - min_val) / (max_val - min_val)
                label = self.class_ids_for_name[name] # Encode labels
                if label == 0:
                    label_coded = np.array([1.,0.], dtype=float)
                else:
                    label_coded = np.array([0.,1.], dtype=float)
                yield video_frames, label_coded #label

