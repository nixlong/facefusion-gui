import base64
import io
import os
import uuid
import time
import threading
from time import sleep
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import cv2
import numpy
import numpy as np
from fastapi import APIRouter
from PIL import Image

from facefusion import logger, process_manager, state_manager
from facefusion.audio import create_empty_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_frame
from facefusion.face_analyser import get_one_face
from facefusion.face_selector import select_faces
from facefusion.filesystem import filter_audio_paths, is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.types import Face, Mask, VisionFrame
from facefusion.vision import (
    detect_frame_orientation,
    extract_vision_mask,
    fit_cover_frame,
    merge_vision_mask,
    obscure_frame,
    read_static_image,
    read_static_images,
    read_video_frame,
    restrict_frame,
    unpack_resolution
)
from facefusion.uis.http_server.models import ApiResponse, PreviewResponse

preview_router = APIRouter()


class PreviewTaskStatus(Enum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class PreviewTaskManager:
    def __init__(self):
        self.current_task: Dict[str, Any] = {
            "status": PreviewTaskStatus.IDLE.value,
            "result": None,
            "error": None,
            "start_time": None,
            "end_time": None
        }
        self.lock = threading.Lock()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.current_task.copy()

    def set_status(self, status: PreviewTaskStatus, result: Any = None, error: str = None):
        with self.lock:
            self.current_task["status"] = status.value
            self.current_task["result"] = result
            self.current_task["error"] = error
            if status == PreviewTaskStatus.RUNNING:
                self.current_task["start_time"] = time.time()
            elif status in [PreviewTaskStatus.SUCCESS, PreviewTaskStatus.FAILED]:
                self.current_task["end_time"] = time.time()

    def reset(self):
        with self.lock:
            self.current_task = {
                "status": PreviewTaskStatus.IDLE.value,
                "result": None,
                "error": None,
                "start_time": None,
                "end_time": None
            }


preview_task_manager = PreviewTaskManager()


def numpy_to_base64(image: np.ndarray) -> Optional[str]:
    if image is None:
        return None
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_preview_frame(reference_vision_frame: VisionFrame,
                         source_vision_frames: List[VisionFrame],
                         source_audio_frame: numpy.ndarray,
                         source_voice_frame: numpy.ndarray,
                         target_vision_frame: VisionFrame,
                         preview_mode: str,
                         preview_resolution: str) -> Optional[numpy.ndarray]:
    target_vision_frame = restrict_frame(target_vision_frame, unpack_resolution(preview_resolution))
    temp_vision_frame = target_vision_frame.copy()
    temp_vision_mask = extract_vision_mask(temp_vision_frame)

    if analyse_frame(target_vision_frame[:, :, :3]):
        if preview_mode == 'frame-by-frame':
            temp_vision_frame = obscure_frame(temp_vision_frame[:, :, :3])
            return numpy.hstack((temp_vision_frame, temp_vision_frame))
        if preview_mode == 'face-by-face':
            target_crop_vision_frame, output_crop_vision_frame = create_face_by_face(
                reference_vision_frame, target_vision_frame[:, :, :3], temp_vision_frame[:, :, :3])
            target_crop_vision_frame = obscure_frame(target_crop_vision_frame)
            output_crop_vision_frame = obscure_frame(output_crop_vision_frame)
            return numpy.hstack((target_crop_vision_frame, output_crop_vision_frame))
        temp_vision_frame = obscure_frame(temp_vision_frame)
        return temp_vision_frame

    for processor_module in get_processors_modules(state_manager.get_item('processors')):
        logger.disable()
        if processor_module.pre_process('preview'):
            logger.enable()
            temp_vision_frame, temp_vision_mask = processor_module.process_frame({
                'reference_vision_frame': reference_vision_frame,
                'source_audio_frame': source_audio_frame,
                'source_voice_frame': source_voice_frame,
                'source_vision_frames': source_vision_frames,
                'target_vision_frame': target_vision_frame[:, :, :3],
                'temp_vision_frame': temp_vision_frame[:, :, :3],
                'temp_vision_mask': temp_vision_mask
            })
        logger.enable()

    temp_vision_frame = prepare_output_frame(target_vision_frame, temp_vision_frame, temp_vision_mask)

    if preview_mode == 'frame-by-frame':
        return numpy.hstack((target_vision_frame, temp_vision_frame))
    if preview_mode == 'face-by-face':
        target_crop_vision_frame, output_crop_vision_frame = create_face_by_face(
            reference_vision_frame, target_vision_frame, temp_vision_frame)
        return numpy.hstack((target_crop_vision_frame, output_crop_vision_frame))

    return temp_vision_frame


def create_face_by_face(reference_vision_frame: VisionFrame,
                        target_vision_frame: VisionFrame,
                        temp_vision_frame: VisionFrame) -> Tuple[VisionFrame, VisionFrame]:
    target_faces = select_faces(reference_vision_frame[:, :, :3], target_vision_frame[:, :, :3])
    target_face = get_one_face(target_faces)

    if target_face:
        target_crop_vision_frame = extract_crop_frame(target_vision_frame, target_face)
        output_crop_vision_frame = extract_crop_frame(temp_vision_frame, target_face)

        if numpy.any(target_crop_vision_frame) and numpy.any(output_crop_vision_frame):
            target_crop_dimension = min(target_crop_vision_frame.shape[:2])
            target_crop_vision_frame = fit_cover_frame(target_crop_vision_frame,
                                                       (target_crop_dimension, target_crop_dimension))
            output_crop_vision_frame = fit_cover_frame(output_crop_vision_frame,
                                                       (target_crop_dimension, target_crop_dimension))
            return target_crop_vision_frame, output_crop_vision_frame

    empty_vision_frame = numpy.zeros((512, 512, 4), dtype=numpy.uint8)
    return empty_vision_frame, empty_vision_frame


def extract_crop_frame(vision_frame: VisionFrame, face: Face) -> Optional[VisionFrame]:
    start_x, start_y, end_x, end_y = map(int, face.bounding_box)
    padding_x = int((end_x - start_x) * 0.25)
    padding_y = int((end_y - start_y) * 0.25)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    crop_vision_frame = vision_frame[start_y:end_y, start_x:end_x]
    return crop_vision_frame


def prepare_output_frame(target_vision_frame: VisionFrame,
                         temp_vision_frame: VisionFrame,
                         temp_vision_mask: Mask) -> VisionFrame:
    temp_vision_mask = temp_vision_mask.clip(
        state_manager.get_item('background_remover_fill_color')[-1], 255)
    temp_vision_frame = merge_vision_mask(temp_vision_frame, temp_vision_mask)
    temp_vision_frame = cv2.resize(temp_vision_frame, target_vision_frame.shape[1::-1])
    return temp_vision_frame


def run_preview_task():
    try:
        preview_task_manager.set_status(PreviewTaskStatus.RUNNING)

        while process_manager.is_checking():
            sleep(0.5)

        source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
        source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
        source_audio_frame = create_empty_audio_frame()
        source_voice_frame = create_empty_audio_frame()

        if source_audio_path and state_manager.get_item('output_video_fps') and state_manager.get_item('reference_frame_number'):
            reference_audio_frame_number = state_manager.get_item('reference_frame_number')
            if state_manager.get_item('trim_frame_start'):
                reference_audio_frame_number -= state_manager.get_item('trim_frame_start')
            temp_voice_frame = get_voice_frame(
                source_audio_path,
                state_manager.get_item('output_video_fps'),
                reference_audio_frame_number
            )
            if numpy.any(temp_voice_frame):
                source_voice_frame = temp_voice_frame

        preview_mode = state_manager.get_item('preview_mode') or 'default'
        preview_resolution = state_manager.get_item('preview_resolution') or '512x512'

        if is_image(state_manager.get_item('target_path')):
            reference_vision_frame = read_static_image(state_manager.get_item('target_path'))
            target_vision_frame = read_static_image(state_manager.get_item('target_path'), 'rgba')
            target_vision_mask = extract_vision_mask(target_vision_frame)
            target_vision_frame = merge_vision_mask(target_vision_frame, target_vision_mask)
            preview_vision_frame = process_preview_frame(
                reference_vision_frame,
                source_vision_frames,
                source_audio_frame,
                source_voice_frame,
                target_vision_frame,
                preview_mode,
                preview_resolution
            )
            output_dir = r'e:\12_facefusion\output_temp'
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f'{uuid.uuid4().hex}.png')
            cv2.imwrite(filepath, preview_vision_frame)
            preview_task_manager.set_status(PreviewTaskStatus.SUCCESS, result=filepath)
            return

        if is_video(state_manager.get_item('target_path')):
            reference_frame_number = state_manager.get_item('reference_frame_number') or 0
            reference_vision_frame = read_video_frame(
                state_manager.get_item('target_path'),
                reference_frame_number
            )
            temp_vision_frame = read_video_frame(
                state_manager.get_item('target_path'),
                reference_frame_number
            )
            temp_vision_mask = extract_vision_mask(temp_vision_frame)
            temp_vision_frame = merge_vision_mask(temp_vision_frame, temp_vision_mask)
            preview_vision_frame = process_preview_frame(
                reference_vision_frame,
                source_vision_frames,
                source_audio_frame,
                source_voice_frame,
                temp_vision_frame,
                preview_mode,
                preview_resolution
            )
            output_dir = r'e:\12_facefusion\output_temp'
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f'{uuid.uuid4().hex}.png')
            cv2.imwrite(filepath, preview_vision_frame)
            preview_task_manager.set_status(PreviewTaskStatus.SUCCESS, result=filepath)
            return

        preview_task_manager.set_status(PreviewTaskStatus.FAILED, error="Unsupported target path type")

    except Exception as e:
        import traceback
        traceback.print_exc()
        preview_task_manager.set_status(PreviewTaskStatus.FAILED, error=str(e))


@preview_router.post("/api/v1/preview/submit", response_model=ApiResponse)
async def submit_preview():
    try:
        task_status = preview_task_manager.get_status()

        if task_status["status"] in [PreviewTaskStatus.PENDING.value, PreviewTaskStatus.RUNNING.value]:
            return ApiResponse(
                success=False,
                message="A preview task is already running"
            )

        source_paths = state_manager.get_item('source_paths')
        target_path = state_manager.get_item('target_path')

        if not source_paths or not target_path:
            return ApiResponse(
                success=False,
                message="source_paths or target_path is not set in state_manager"
            )

        preview_task_manager.reset()
        preview_task_manager.set_status(PreviewTaskStatus.PENDING)

        thread = threading.Thread(target=run_preview_task)
        thread.daemon = True
        thread.start()

        return ApiResponse(
            success=True,
            message="Preview task submitted successfully"
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=str(e)
        )


@preview_router.get("/api/v1/preview/status")
async def get_preview_status():
    try:
        task_status = preview_task_manager.get_status()

        response_data = {
            "status": task_status["status"],
            "start_time": task_status["start_time"],
            "end_time": task_status["end_time"],
            "result": None
        }

        if task_status["status"] == PreviewTaskStatus.SUCCESS.value:
            response_data["result"] = task_status["result"]
        elif task_status["status"] == PreviewTaskStatus.FAILED.value:
            response_data["error"] = task_status["error"]

        return ApiResponse(
            success=True,
            data=response_data,
            message="Preview status retrieved successfully"
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            message=str(e)
        )

