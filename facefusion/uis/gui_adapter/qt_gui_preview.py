import cv2
import numpy
import time
import threading
from typing import List, Optional, Dict, Any
from enum import Enum

from facefusion import state_manager, process_manager
from facefusion.audio import create_empty_audio_frame, get_voice_frame
from facefusion.filesystem import filter_audio_paths, is_image, is_video
from facefusion.vision import read_static_images, read_static_image, read_video_frame, extract_vision_mask, merge_vision_mask, detect_frame_orientation
from facefusion.processors.core import get_processors
from facefusion.processors.modules.face_swapper import core as face_swapper_core


class PreviewStatus(Enum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class PreviewTask:
    def __init__(self):
        self.status: str = PreviewStatus.IDLE.value
        self.result: Optional[numpy.ndarray] = None
        self.error: Optional[str] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class PreviewStateManager:
    def __init__(self):
        self.current_task: PreviewTask = PreviewTask()
        self.lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.current_task.to_dict()

    def set_status(self, status: PreviewStatus, result: Any = None, error: str = None):
        with self.lock:
            self.current_task.status = status.value
            self.current_task.result = result
            self.current_task.error = error
            if status == PreviewStatus.RUNNING:
                self.current_task.start_time = time.time()
            elif status in [PreviewStatus.SUCCESS, PreviewStatus.FAILED]:
                self.current_task.end_time = time.time()

    def reset(self):
        with self.lock:
            self.current_task = PreviewTask()

    def update_progress(self, progress: int):
        pass


preview_state_manager = PreviewStateManager()


class QtGuiPreview:
    def __init__(self):
        self.preview_status = {
            "status": "idle",
            "progress": 0,
            "error": None,
            "result": None
        }

    def update_preview_status(self, status: str, progress: int = 0, error: Optional[str] = None, result: Optional[numpy.ndarray] = None):
        self.preview_status = {
            "status": status,
            "progress": progress,
            "error": error,
            "result": result
        }

    def get_preview_status(self) -> dict:
        return self.preview_status

    def process_preview_frame(self, reference_vision_frame: numpy.ndarray, source_vision_frames: List[numpy.ndarray],
                            source_audio_frame: numpy.ndarray, source_voice_frame: numpy.ndarray,
                            target_vision_frame: numpy.ndarray, preview_mode: str, preview_resolution: str) -> numpy.ndarray:
        try:
            self.update_preview_status("processing", 0)
            time.sleep(1)
            preview_vision_frame = target_vision_frame.copy()
            self.update_preview_status("completed", 100, result=preview_vision_frame)
            return preview_vision_frame
        except Exception as e:
            self.update_preview_status("error", 0, error=str(e))
            return target_vision_frame

    def update_preview_image(self, preview_mode: str, preview_resolution: str, frame_number: int = 0) -> Optional[numpy.ndarray]:
        self.update_preview_status("processing", 0)

        while process_manager.is_checking():
            time.sleep(0.5)

        source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
        source_audio_path = next(iter(filter_audio_paths(state_manager.get_item('source_paths'))), None)
        source_audio_frame = create_empty_audio_frame()
        source_voice_frame = create_empty_audio_frame()

        if source_audio_path and state_manager.get_item('output_video_fps') and state_manager.get_item('reference_frame_number'):
            reference_audio_frame_number = state_manager.get_item('reference_frame_number')
            if state_manager.get_item('trim_frame_start'):
                reference_audio_frame_number -= state_manager.get_item('trim_frame_start')
            temp_voice_frame = get_voice_frame(source_audio_path, state_manager.get_item('output_video_fps'), reference_audio_frame_number)
            if numpy.any(temp_voice_frame):
                source_voice_frame = temp_voice_frame

        if is_image(state_manager.get_item('target_path')):
            reference_vision_frame = read_static_image(state_manager.get_item('target_path'))
            target_vision_frame = read_static_image(state_manager.get_item('target_path'), 'rgba')
            target_vision_mask = extract_vision_mask(target_vision_frame)
            target_vision_frame = merge_vision_mask(target_vision_frame, target_vision_mask)
            preview_vision_frame = self.process_preview_frame(reference_vision_frame, source_vision_frames,
                                                           source_audio_frame, source_voice_frame,
                                                           target_vision_frame, preview_mode, preview_resolution)
            preview_vision_frame = cv2.cvtColor(preview_vision_frame, cv2.COLOR_BGRA2RGBA)
            return preview_vision_frame

        if is_video(state_manager.get_item('target_path')):
            reference_vision_frame = read_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
            temp_vision_frame = read_video_frame(state_manager.get_item('target_path'), frame_number)
            temp_vision_mask = extract_vision_mask(temp_vision_frame)
            temp_vision_frame = merge_vision_mask(temp_vision_frame, temp_vision_mask)
            preview_vision_frame = self.process_preview_frame(reference_vision_frame, source_vision_frames,
                                                           source_audio_frame, source_voice_frame,
                                                           temp_vision_frame, preview_mode, preview_resolution)
            preview_vision_frame = cv2.cvtColor(preview_vision_frame, cv2.COLOR_BGRA2RGBA)
            return preview_vision_frame

        self.update_preview_status("completed", 100, result=None)
        return None


qt_gui_preview = QtGuiPreview()


def run_preview_in_thread(preview_mode: str, preview_resolution: str, frame_number: int = 0):
    try:
        preview_state_manager.set_status(PreviewStatus.RUNNING)
        result = qt_gui_preview.update_preview_image(preview_mode, preview_resolution, frame_number)
        preview_state_manager.set_status(PreviewStatus.SUCCESS, result=result)
    except Exception as e:
        preview_state_manager.set_status(PreviewStatus.FAILED, error=str(e))


def handle_preview_submit(preview_mode: str, preview_resolution: str, frame_number: int = 0) -> dict:
    try:
        task_status = preview_state_manager.get_status()

        if task_status["status"] in [PreviewStatus.PENDING.value, PreviewStatus.RUNNING.value]:
            return {
                "success": False,
                "message": "A preview task is already running"
            }

        preview_state_manager.reset()
        preview_state_manager.set_status(PreviewStatus.PENDING)

        thread = threading.Thread(
            target=run_preview_in_thread,
            args=(preview_mode, preview_resolution, frame_number)
        )
        thread.daemon = True
        thread.start()

        return {
            "success": True,
            "message": "Preview task submitted successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def handle_preview_status() -> dict:
    try:
        task_status = preview_state_manager.get_status()

        response_data = {
            "status": task_status["status"],
            "start_time": task_status["start_time"],
            "end_time": task_status["end_time"]
        }

        if task_status["status"] == PreviewStatus.SUCCESS.value:
            response_data["result"] = task_status["result"]
        elif task_status["status"] == PreviewStatus.FAILED.value:
            response_data["error"] = task_status["error"]

        return {
            "success": True,
            "data": response_data,
            "message": "Preview status retrieved successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def handle_preview_request(preview_mode: str, preview_resolution: str, frame_number: int = 0) -> dict:
    try:
        result = qt_gui_preview.update_preview_image(preview_mode, preview_resolution, frame_number)
        status = qt_gui_preview.get_preview_status()
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }