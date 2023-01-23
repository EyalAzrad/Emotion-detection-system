import cv2

class VideoManager:
    def __init__(self):
        """
        Initialize an empty list to store the frames of each clip
        """
        self.frames = []

    def process_video(self, video_path, seconds_per_clip, dst=None):
        """
            Split a video into clips and save each clip's frames to a directory.

            Parameters
            ----------
            video_path : str
                Path of the video to process.
            seconds_per_clip : float
                The duration of each clip in seconds.
            dst : Optional[str], optional
                The directory to save the frames of each clip, by default None.

            Returns
            -------
            None
        """
        vidcap = cv2.VideoCapture(video_path)
        frames_per_clip = int(seconds_per_clip * vidcap.get(cv2.CAP_PROP_FPS))
        success, image = vidcap.read()
        clip_frames = []
        count = 0
        while success:
            if count % 3 == 0:
                clip_frames.append(image)
            success, image = vidcap.read()
            count += 1
            if count % frames_per_clip == 0:
                self.frames.append(clip_frames)
                clip_frames = []

        if dst is not None:
            for i, clip in enumerate(self.frames):
                for j, frame in enumerate(clip):
                    destination = dst + "//" + f"frame_{i}_{j}.jpg"
                    cv2.imwrite(destination, frame)
