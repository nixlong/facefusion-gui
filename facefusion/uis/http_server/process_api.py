import threading
import time
import uuid
from typing import Optional

from fastapi import APIRouter

from facefusion import state_manager
from facefusion.processors.core import get_processors_modules
from facefusion.workflows.image_to_image import process as image_to_image_process
from facefusion.workflows.image_to_video import process as image_to_video_process
from facefusion.filesystem import is_image, is_video
from facefusion.uis.http_server.models import ApiResponse, ProcessRequest, ProcessResponse, ProcessStatusResponse, ProgressResponse

process_router = APIRouter()

_process_status = {
    "status": "idle",
    "task_id": None,
    "progress": 0,
    "message": "",
    "current": 0,
    "total": 0,
    "stop_event": None
}
_status_lock = threading.Lock()


def update_status(status: str, task_id: Optional[str] = None, progress: int = 0,
                 message: str = "", current: int = 0, total: int = 0):
    with _status_lock:
        _process_status["status"] = status
        if task_id is not None:
            _process_status["task_id"] = task_id
        _process_status["progress"] = progress
        _process_status["message"] = message
        _process_status["current"] = current
        _process_status["total"] = total


def process_image(source_paths, target_path, processors, output_path):
    update_status("running", message="Processing image...")

    try:
        for processor_module in get_processors_modules(processors):
            if not processor_module.pre_process("output"):
                update_status("failed", message="Pre-processing failed")
                return

        error_code = image_to_image_process(time.time())
        if error_code == 0:
            update_status("completed", progress=100, message="Processing completed")
        else:
            update_status("failed", message=f"Processing failed with code {error_code}")
    except Exception as e:
        update_status("failed", message=str(e))


def process_video(source_paths, target_path, processors, output_path, fps):
    update_status("running", message="Processing video...")

    try:
        for processor_module in get_processors_modules(processors):
            if not processor_module.pre_process("output"):
                update_status("failed", message="Pre-processing failed")
                return

        error_code = image_to_video_process(time.time())
        if error_code == 0:
            update_status("completed", progress=100, message="Processing completed")
        else:
            update_status("failed", message=f"Processing failed with code {error_code}")
    except Exception as e:
        update_status("failed", message=str(e))


@process_router.post("/api/v1/process/start", response_model=ProcessResponse)
async def start_process(request: ProcessRequest):
    try:
        with _status_lock:
            if _process_status["status"] == "running":
                return ProcessResponse(
                    success=False,
                    message="Another process is already running"
                )

        task_id = f"process_{uuid.uuid4().hex[:12]}"

        state_manager.set_item("source_paths", request.source_paths)
        state_manager.set_item("target_path", request.target_path)
        state_manager.set_item("processors", request.processors)
        state_manager.set_item("output_path", request.output_path)

        update_status("queued", task_id=task_id, message="Task queued")

        if is_image(request.target_path):
            thread = threading.Thread(
                target=process_image,
                args=(request.source_paths, request.target_path,
                      request.processors, request.output_path)
            )
        elif is_video(request.target_path):
            thread = threading.Thread(
                target=process_video,
                args=(request.source_paths, request.target_path,
                      request.processors, request.output_path, request.output_video_fps)
            )
        else:
            return ProcessResponse(success=False, message="Unsupported target file type")

        thread.start()

        return ProcessResponse(
            success=True,
            task_id=task_id,
            message="Processing started"
        )
    except Exception as e:
        return ProcessResponse(success=False, message=str(e))


@process_router.post("/api/v1/process/stop", response_model=ApiResponse)
async def stop_process():
    try:
        with _status_lock:
            if _process_status["status"] != "running":
                return ApiResponse(success=False, message="No process is running")

            if _process_status.get("stop_event"):
                _process_status["stop_event"].set()

        update_status("idle", message="Process stopped by user")
        return ApiResponse(success=True, message="Process stopped")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@process_router.get("/api/v1/process/status", response_model=ProcessStatusResponse)
async def get_process_status():
    try:
        with _status_lock:
            return ProcessStatusResponse(
                success=True,
                status=_process_status["status"],
                task_id=_process_status.get("task_id"),
                progress=_process_status.get("progress", 0),
                message=_process_status.get("message", "")
            )
    except Exception as e:
        return ProcessStatusResponse(success=False, status="error", message=str(e))


@process_router.get("/api/v1/process/progress", response_model=ProgressResponse)
async def get_process_progress():
    try:
        with _status_lock:
            return ProgressResponse(
                success=True,
                current=_process_status.get("current", 0),
                total=_process_status.get("total", 0),
                percentage=_process_status.get("progress", 0)
            )
    except Exception as e:
        return ProgressResponse(success=False, current=0, total=0, percentage=0.0, message=str(e))
