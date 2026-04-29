import cv2
import numpy as np
from facefusion import state_manager
from facefusion.filesystem import is_image, is_video
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_and_filter_faces
from facefusion.uis.components.face_selector import extract_gallery_frames
from facefusion.vision import fit_cover_frame, read_static_image, read_video_frame


def extract_reference_faces():
    """
    从目标文件中提取参考人脸
    返回: 参考人脸列表，每个元素是 RGB 格式的 numpy 数组
    """
    target_path = state_manager.get_item('target_path')
    if not target_path:
        return []

    gallery_vision_frames = []
    if is_image(target_path):
        target_vision_frame = read_static_image(target_path)
        gallery_vision_frames = extract_gallery_frames(target_vision_frame)
    elif is_video(target_path):
        reference_frame_number = state_manager.get_item('reference_frame_number', 0)
        target_vision_frame = read_video_frame(target_path, reference_frame_number)
        gallery_vision_frames = extract_gallery_frames(target_vision_frame)

    return gallery_vision_frames


def _extract_gallery_frames(target_vision_frame):
    """
    从目标帧中提取人脸区域
    参考 face_selector.py 的实现
    """
    gallery_vision_frames = []
    faces = get_many_faces([target_vision_frame])
    faces = sort_and_filter_faces(faces)

    for face in faces:
        start_x, start_y, end_x, end_y = map(int, face.bounding_box)
        padding_x = int((end_x - start_x) * 0.25)
        padding_y = int((end_y - start_y) * 0.25)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        crop_vision_frame = target_vision_frame[start_y:end_y, start_x:end_x]
        crop_vision_frame = fit_cover_frame(crop_vision_frame, (128, 128))
        crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_BGR2RGB)
        gallery_vision_frames.append(crop_vision_frame)
    return gallery_vision_frames


def get_reference_face_count():
    """
    获取目标文件中的人脸数量
    """
    faces = extract_reference_faces()
    return len(faces)


def get_reference_face_by_index(index):
    """
    根据索引获取参考人脸
    """
    faces = extract_reference_faces()
    if 0 <= index < len(faces):
        return faces[index]
    return None