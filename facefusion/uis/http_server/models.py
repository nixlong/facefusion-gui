from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    success: bool
    status: str
    uptime: float


class InfoResponse(BaseModel):
    success: bool
    version: str
    python_version: str
    platform: str


class StateItemRequest(BaseModel):
    key: str
    value: Any


class StateBatchRequest(BaseModel):
    source_paths: Optional[List[str]] = None
    target_path: Optional[str] = None
    processors: Optional[List[str]] = None
    execution_providers: Optional[List[str]] = None
    preview_resolution: Optional[str] = None
    preview_mode: Optional[str] = None


class PreviewRequest(BaseModel):
    source_paths: List[str]
    target_path: str
    processors: List[str]
    preview_mode: str = "default"
    preview_resolution: str = "512x512"


class PreviewResponse(BaseModel):
    success: bool
    frame: Optional[str] = None
    frame_shape: Optional[List[int]] = None
    processing_time: Optional[float] = None
    message: Optional[str] = None


class ProcessRequest(BaseModel):
    source_paths: List[str]
    target_path: str
    processors: List[str]
    output_path: str
    output_video_fps: Optional[int] = None


class ProcessResponse(BaseModel):
    success: bool
    task_id: Optional[str] = None
    message: Optional[str] = None


class ProcessStatusResponse(BaseModel):
    success: bool
    status: str
    task_id: Optional[str] = None
    progress: Optional[int] = None
    message: Optional[str] = None


class ProgressResponse(BaseModel):
    success: bool
    current: int
    total: int
    percentage: float
    eta: Optional[str] = None
