import base64
import io
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter
from PIL import Image

from facefusion import state_manager
from facefusion.content_analyser import analyse_frame
from facefusion.filesystem import is_image, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.vision import read_static_image, read_static_images, read_video_frame, restrict_frame, unpack_resolution
from facefusion.uis.http_server.models import ApiResponse, PreviewRequest, PreviewResponse

preview_router = APIRouter()


def numpy_to_base64(image: np.ndarray) -> str:
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


def process_preview_frame(source_paths: List[str], target_path: str,
                         processors: List[str], preview_mode: str,
                         preview_resolution: str) -> Optional[str]:
    try:
        state_manager.set_item("source_paths", source_paths)
        state_manager.set_item("target_path", target_path)
        state_manager.set_item("processors", processors)

        source_vision_frames = read_static_images(source_paths)

        if is_image(target_path):
            reference_vision_frame = read_static_image(target_path)
            target_vision_frame = read_static_image(target_path, "rgba")
        elif is_video(target_path):
            reference_frame_number = state_manager.get_item("reference_frame_number") or 0
            reference_vision_frame = read_video_frame(target_path, reference_frame_number)
            target_vision_frame = read_video_frame(target_path, reference_frame_number)
        else:
            return None

        target_vision_frame = restrict_frame(target_vision_frame, unpack_resolution(preview_resolution))
        temp_vision_frame = target_vision_frame.copy()

        if analyse_frame(target_vision_frame[:, :, :3]):
            return numpy_to_base64(np.zeros((512, 512, 3), dtype=np.uint8))

        for processor_module in get_processors_modules(processors):
            if processor_module.pre_process("preview"):
                temp_vision_frame, temp_vision_mask = processor_module.process_frame({
                    "reference_vision_frame": reference_vision_frame,
                    "source_vision_frames": source_vision_frames,
                    "target_vision_frame": target_vision_frame[:, :, :3],
                    "temp_vision_frame": temp_vision_frame[:, :, :3],
                    "temp_vision_mask": None
                })

        return numpy_to_base64(temp_vision_frame)
    except Exception as e:
        print(f"[PreviewAPI] process_preview_frame error: {e}")
        import traceback
        traceback.print_exc()
        return None


@preview_router.post("/api/v1/preview", response_model=PreviewResponse)
async def preview(request: PreviewRequest):
    start_time = time.time()
    try:
        frame_base64 = process_preview_frame(
            source_paths=request.source_paths,
            target_path=request.target_path,
            processors=request.processors,
            preview_mode=request.preview_mode,
            preview_resolution=request.preview_resolution
        )

        if frame_base64 is None:
            return PreviewResponse(
                success=False,
                message="Preview processing failed"
            )

        processing_time = time.time() - start_time
        return PreviewResponse(
            success=True,
            frame=frame_base64,
            processing_time=processing_time,
            message="Preview generated successfully"
        )
    except Exception as e:
        return PreviewResponse(
            success=False,
            message=str(e)
        )


@preview_router.post("/api/v1/preview/quick", response_model=PreviewResponse)
async def quick_preview(request: PreviewRequest):
    start_time = time.time()
    try:
        frame_base64 = process_preview_frame(
            source_paths=request.source_paths,
            target_path=request.target_path,
            processors=request.processors,
            preview_mode=request.preview_mode,
            preview_resolution="256x256"
        )

        if frame_base64 is None:
            return PreviewResponse(
                success=False,
                message="Quick preview processing failed"
            )

        processing_time = time.time() - start_time
        return PreviewResponse(
            success=True,
            frame=frame_base64,
            processing_time=processing_time,
            message="Quick preview generated successfully"
        )
    except Exception as e:
        return PreviewResponse(
            success=False,
            message=str(e)
        )
