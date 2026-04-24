from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class FaceFusionStateItems(BaseModel):
    """
    参照 UpdateConfigRequest 规范
    所有参数均为可选，支持单独修改任意一个参数
    自动类型校验，适配 FastAPI 接口
    """
    # 基础参数
    age_modifier_direction: Optional[int] = None
    age_modifier_model: Optional[str] = None
    background_remover_despill_color: Optional[List[int]] = None
    background_remover_fill_color: Optional[List[int]] = None
    background_remover_model: Optional[str] = None
    benchmark_cycle_count: Optional[int] = None
    benchmark_mode: Optional[str] = None
    benchmark_resolutions: Optional[List[str]] = None
    command: Optional[str] = None
    config_path: Optional[str] = None
    
    # 深度换脸
    deep_swapper_model: Optional[str] = None
    deep_swapper_morph: Optional[int] = None
    
    # 下载/执行配置
    download_providers: Optional[List[str]] = None
    execution_device_ids: Optional[List[int]] = None
    execution_providers: Optional[List[str]] = None
    execution_thread_count: Optional[int] = None
    
    # 表情修复
    expression_restorer_areas: Optional[List[str]] = None
    expression_restorer_factor: Optional[int] = None
    expression_restorer_model: Optional[str] = None
    
    # 人脸调试
    face_debugger_items: Optional[List[str]] = None
    
    # 人脸检测
    face_detector_angles: Optional[List[int]] = None
    face_detector_margin: Optional[List[int]] = None
    face_detector_model: Optional[str] = None
    face_detector_score: Optional[float] = None
    face_detector_size: Optional[str] = None
    
    # 人脸编辑器
    face_editor_eye_gaze_horizontal: Optional[float] = None
    face_editor_eye_gaze_vertical: Optional[float] = None
    face_editor_eye_open_ratio: Optional[float] = None
    face_editor_eyebrow_direction: Optional[float] = None
    face_editor_head_pitch: Optional[float] = None
    face_editor_head_roll: Optional[float] = None
    face_editor_head_yaw: Optional[float] = None
    face_editor_lip_open_ratio: Optional[float] = None
    face_editor_model: Optional[str] = None
    face_editor_mouth_grim: Optional[float] = None
    face_editor_mouth_position_horizontal: Optional[float] = None
    face_editor_mouth_position_vertical: Optional[float] = None
    face_editor_mouth_pout: Optional[float] = None
    face_editor_mouth_purse: Optional[float] = None
    face_editor_mouth_smile: Optional[float] = None
    
    # 人脸增强
    face_enhancer_blend: Optional[int] = None
    face_enhancer_model: Optional[str] = None
    face_enhancer_weight: Optional[float] = None
    
    # 人脸关键点
    face_landmarker_model: Optional[str] = None
    face_landmarker_score: Optional[float] = None
    
    # 人脸遮罩
    face_mask_areas: Optional[List[str]] = None
    face_mask_blur: Optional[float] = None
    face_mask_padding: Optional[List[int]] = None
    face_mask_regions: Optional[List[str]] = None
    face_mask_types: Optional[List[str]] = None
    
    # 人脸遮挡/解析
    face_occluder_model: Optional[str] = None
    face_parser_model: Optional[str] = None
    
    # 人脸选择
    face_selector_age_end: Optional[int] = None
    face_selector_age_start: Optional[int] = None
    face_selector_gender: Optional[str] = None
    face_selector_mode: Optional[str] = None
    face_selector_order: Optional[str] = None
    face_selector_race: Optional[str] = None
    
    # 人脸交换
    face_swapper_model: Optional[str] = None
    face_swapper_pixel_boost: Optional[str] = None
    face_swapper_weight: Optional[float] = None
    
    # 帧处理
    frame_colorizer_blend: Optional[int] = None
    frame_colorizer_model: Optional[str] = None
    frame_colorizer_size: Optional[str] = None
    frame_enhancer_blend: Optional[int] = None
    frame_enhancer_model: Optional[str] = None
    
    # 路径/日志
    jobs_path: Optional[str] = None
    keep_temp: Optional[bool] = None
    lip_syncer_model: Optional[str] = None
    lip_syncer_weight: Optional[float] = None
    log_level: Optional[str] = None
    open_browser: Optional[bool] = None
    
    # 输出配置
    output_audio_encoder: Optional[str] = None
    output_audio_quality: Optional[int] = None
    output_audio_volume: Optional[int] = None
    output_image_quality: Optional[int] = None
    output_image_scale: Optional[float] = None
    output_path: Optional[str] = None
    output_video_encoder: Optional[str] = None
    output_video_fps: Optional[int] = None
    output_video_preset: Optional[str] = None
    output_video_quality: Optional[int] = None
    output_video_scale: Optional[float] = None
    
    # 处理核心
    processors: Optional[List[str]] = None
    reference_face_distance: Optional[float] = None
    reference_face_position: Optional[int] = None
    reference_frame_number: Optional[int] = None
    
    # 源/目标路径
    source_paths: Optional[List[str]] = None
    system_memory_limit: Optional[int] = None
    target_path: Optional[str] = None
    
    # 临时文件
    temp_frame_format: Optional[str] = None
    temp_path: Optional[str] = None
    
    # 裁剪
    trim_frame_end: Optional[int] = None
    trim_frame_start: Optional[int] = None
    
    # UI配置
    ui_layouts: Optional[List[str]] = None
    ui_workflow: Optional[str] = None
    
    # 系统策略
    video_memory_strategy: Optional[str] = None
    voice_extractor_model: Optional[str] = None

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
