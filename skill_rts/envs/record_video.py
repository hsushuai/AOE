import gym
import os
import numpy as np
from PIL import Image
from typing import Callable
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from skill_rts import logger


class RecordVideo(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str = "videos",
        name_prefix: str = None,
        record_trigger: Callable[[int], bool] = lambda x: True,
        render_fps = 150,
        show: bool = True,
        theme: str="white"
    ):
        super().__init__(env)
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)
        
        self.name_prefix = name_prefix
        self.record_trigger = record_trigger
        self.render_mode = "rgb_array"  # only rgb_array mode supported
        self.render_fps = render_fps
        self.is_show = show
        self.theme = theme

        self.recording = False
        self.recorded_frames = 0
        self.step_id = 0
        self.video_recorder: VideoRecorder = None

        self.metadata={"render_modes": [self.render_mode], "render_fps": self.render_fps}
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.step_id = 0
        if self._video_enable():
            self.start_video_recorder()
        else:
            self.close_video_recorder()
        return obs

    def step(self, action):
        obs, rewards, dones, infos = super().step(action)

        self.step_id += 1
        self.recorded_frames += 1

        if self.recording:
            self.render()
            self.video_recorder.capture_frame()
            if dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return obs, rewards, dones, infos
    
    def render(self, mode: str="rgb_array"):
        bytes_array = np.array(self.render_client.render(self.is_show, self.theme))
        image = Image.frombytes("RGB", (640, 640), bytes_array)
        return np.array(image)[:, :, ::-1]
    
    def _video_enable(self):
        return self.record_trigger(self.step_id)
    
    def start_video_recorder(self):
        self.close_video_recorder()

        filename = self._filename()
        self.video_recorder = VideoRecorder(self, path=filename)
        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True
    
    def _filename(self):
        filename = f"{self.name_prefix}-video" if self.name_prefix else "video"
        filename = os.path.join(self.video_folder, filename)
        return f"{filename}.mp4"
    
    def close_video_recorder(self) -> None:
        if self.recording and self.video_recorder:
            self.video_recorder.close()
        self.recording = False
    
    def close(self):
        super().close()
        self.close_video_recorder()
    
    def __del__(self):
        self.close()
