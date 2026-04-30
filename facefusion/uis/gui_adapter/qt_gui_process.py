import os
import time
import threading
import gc
from typing import Dict, Any, Optional
from enum import Enum


class ProcessStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ProcessTaskManager:
    def __init__(self):
        self.current_task: Dict[str, Any] = {
            "status": ProcessStatus.IDLE.value,
            "progress": 0,
            "message": "",
            "error": None,
            "result": None,
            "start_time": None,
            "end_time": None
        }
        self.lock = threading.Lock()
        self.progress_callback = None
        self.status_callback = None
        self._is_stopped = False

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.current_task.copy()

    def set_status(self, status: ProcessStatus, message: str = "", error: str = None, result: Any = None):
        with self.lock:
            self.current_task["status"] = status.value
            self.current_task["message"] = message
            self.current_task["error"] = error
            self.current_task["result"] = result
            if status == ProcessStatus.RUNNING:
                self.current_task["start_time"] = time.time()
            elif status in [ProcessStatus.COMPLETED, ProcessStatus.FAILED, ProcessStatus.STOPPED]:
                self.current_task["end_time"] = time.time()
        
        print(f"[ProcessTaskManager] 状态更新: {status.value}, 消息: {message}, 错误: {error}, 结果: {result}")
        
        if self.status_callback:
            print(f"[ProcessTaskManager] 调用状态回调")
            try:
                self.status_callback(self.current_task.copy())
            except Exception as e:
                print(f"[ProcessTaskManager] 状态回调调用失败: {e}")

    def update_progress(self, progress: int, message: str = ""):
        with self.lock:
            if self._is_stopped:
                print(f"[ProcessTaskManager] 任务已停止，跳过进度更新")
                return
            self.current_task["progress"] = progress
            self.current_task["message"] = message
        
        print(f"[ProcessTaskManager] 进度更新: {progress}%, 消息: {message}")
        
        if self.progress_callback:
            try:
                self.progress_callback(progress, message)
            except Exception as e:
                print(f"[ProcessTaskManager] 进度回调调用失败: {e}")

    def reset(self):
        print("[ProcessTaskManager] 重置任务状态")
        with self.lock:
            self.current_task = {
                "status": ProcessStatus.IDLE.value,
                "progress": 0,
                "message": "",
                "error": None,
                "result": None,
                "start_time": None,
                "end_time": None
            }
            self._is_stopped = False

    def stop(self):
        print("[ProcessTaskManager] 收到停止请求")
        with self.lock:
            self._is_stopped = True
        print("[ProcessTaskManager] 调用 process_manager.stop()")
        from facefusion import process_manager
        process_manager.stop()
        print("[ProcessTaskManager] 停止请求处理完成")


class TqdmProgressBridge:
    STAGE_WEIGHTS = {
        'analysing': (0, 25),
        'extracting': (25, 50),
        'processing': (50, 90),
        'merging': (90, 100),
    }

    _original_tqdm_update = None
    _active = False

    @classmethod
    def install(cls):
        if cls._active:
            return
        from tqdm import tqdm
        cls._original_tqdm_update = tqdm.update
        tqdm.update = cls._patched_update
        cls._active = True
        print("[TqdmProgressBridge] 已安装 tqdm 进度拦截")

    @classmethod
    def uninstall(cls):
        if not cls._active or cls._original_tqdm_update is None:
            return
        from tqdm import tqdm
        tqdm.update = cls._original_tqdm_update
        cls._original_tqdm_update = None
        cls._active = False
        print("[TqdmProgressBridge] 已卸载 tqdm 进度拦截")

    @staticmethod
    def _patched_update(tqdm_self, n=1):
        TqdmProgressBridge._original_tqdm_update(tqdm_self, n)

        if tqdm_self.disable or not tqdm_self.total:
            return

        desc = tqdm_self.desc or ''
        matched_stage = None
        for stage_key in TqdmProgressBridge.STAGE_WEIGHTS:
            if stage_key in desc.lower():
                matched_stage = stage_key
                break

        if matched_stage:
            min_pct, max_pct = TqdmProgressBridge.STAGE_WEIGHTS[matched_stage]
            stage_progress = int(tqdm_self.n / tqdm_self.total * 100) if tqdm_self.total > 0 else 0
            overall_pct = min_pct + int(stage_progress * (max_pct - min_pct) / 100)
            overall_pct = min(max(overall_pct, min_pct), max_pct)

            process_task_manager.update_progress(
                overall_pct,
                f"{desc}: {tqdm_self.n}/{tqdm_self.total}"
            )


process_task_manager = ProcessTaskManager()


def clear_process_environment():
    print("[Process] 开始清理执行环境")
    try:
        from facefusion import state_manager
        from facefusion.temp_helper import clear_temp_directory
        
        target_path = state_manager.get_item('target_path')
        if target_path:
            clear_temp_directory(target_path)
            print(f"[Process] 已清理临时目录: {target_path}")
        
        from facefusion.vision import read_static_image, read_static_video_frame
        read_static_image.cache_clear()
        read_static_video_frame.cache_clear()
        print("[Process] 已清理图像缓存")
        
        gc.collect()
        print("[Process] 已执行垃圾回收")
        
        print("[Process] 执行环境清理完成")
    except Exception as e:
        print(f"[Process] 清理环境时出错: {e}")

def on_progress(progress: int, message: str):
    print(f"[Process] 收到进度更新: {progress}% - {message}")
    process_task_manager.update_progress(progress, message)

def create_and_run_job(step_args: Dict[str, Any]) -> bool:
    print(f"[Process] 创建并运行任务，参数: {step_args}")
    
    from facefusion.jobs import job_helper, job_manager, job_runner, job_store
    from facefusion import state_manager
    
    job_id = job_helper.suggest_job_id('ui')
    print(f"[Process] 生成的任务ID: {job_id}")
    
    print("[Process] 同步状态管理器项目")
    for key in job_store.get_job_keys():
        state_manager.sync_item(key)
    
    print("[Process] 创建任务...")
    create_ok = job_manager.create_job(job_id)
    print(f"[Process] 创建任务结果: {create_ok}")
    
    if not create_ok:
        return False
    
    print("[Process] 添加任务步骤...")
    add_step_ok = job_manager.add_step(job_id, step_args)
    print(f"[Process] 添加步骤结果: {add_step_ok}")
    
    if not add_step_ok:
        return False
    
    print("[Process] 提交任务...")
    submit_ok = job_manager.submit_job(job_id)
    print(f"[Process] 提交任务结果: {submit_ok}")
    
    if not submit_ok:
        return False
    
    print("[Process] 运行任务...")
    from facefusion.core import process_step
    run_ok = job_runner.run_job(job_id, process_step)
    print(f"[Process] 运行任务结果: {run_ok}")
    
    return run_ok





def run_process_task():
    print("[Process] 开始执行合成任务")
    TqdmProgressBridge.install()
    try:
        print("[Process] 设置状态为 RUNNING")
        process_task_manager.set_status(ProcessStatus.RUNNING, "开始处理...")
        
        print("[Process] 导入必要模块...")
        from facefusion.args import collect_step_args
        
        from facefusion.filesystem import is_directory
        from facefusion.uis.ui_helper import suggest_output_path
        from facefusion.jobs import job_manager
        from facefusion import state_manager
        print("[Process] 模块导入成功")
        
        print("[Process] 收集步骤参数...")
        step_args = collect_step_args()
        print(f"[Process] 收集到的参数: {step_args}")
        
        print("[Process] 获取输出路径...")
        output_path = state_manager.get_item('output_path')
        print(f"[Process] 当前输出路径: {output_path}")
        
        if output_path:
            if is_directory(output_path):
                output_path = suggest_output_path(output_path, state_manager.get_item('target_path'))
                print(f"[Process] 自动生成输出路径: {output_path}")
            step_args['output_path'] = output_path
        
        print("[Process] 初始化任务管理...")
        jobs_path = state_manager.get_item('jobs_path')
        print(f"[Process] jobs_path: {jobs_path}")
        
        if job_manager.init_jobs(jobs_path):
            print("[Process] 任务管理初始化成功")
            process_task_manager.update_progress(10, "初始化任务...")
            
            print("[Process] 创建并运行任务...")
            success = create_and_run_job(step_args)
            
            print(f"[Process] 任务执行完成，结果: {success}")
            
            if process_task_manager._is_stopped:
                print("[Process] 检测到任务已被停止")
                process_task_manager.set_status(
                    ProcessStatus.STOPPED,
                    "任务已停止"
                )
                clear_process_environment()
                return
            
            if success:
                print("[Process] 任务执行成功")
                process_task_manager.set_status(
                    ProcessStatus.COMPLETED, 
                    "处理完成",
                    result=step_args.get('output_path')
                )
            else:
                print("[Process] 任务执行失败")
                process_task_manager.set_status(
                    ProcessStatus.FAILED,
                    "任务执行失败",
                    error="任务执行失败"
                )
        else:
            print("[Process] 任务管理初始化失败")
            process_task_manager.set_status(
                ProcessStatus.FAILED,
                "任务管理初始化失败",
                error="无法初始化任务管理"
            )
            
    except Exception as e:
        import traceback
        print(f"[Process] 执行任务时发生异常: {e}")
        traceback.print_exc()
        if process_task_manager._is_stopped:
            process_task_manager.set_status(
                ProcessStatus.STOPPED,
                "任务已停止"
            )
        else:
            process_task_manager.set_status(
                ProcessStatus.FAILED,
                "处理异常",
                error=str(e)
            )
    finally:
        print("[Process] 执行 finally 清理")
        TqdmProgressBridge.uninstall()
        clear_process_environment()
        print("[Process] 合成任务执行结束")


def start_process_task(progress_callback=None, status_callback=None):
    print(f"[Process] start_process_task 被调用, progress_callback: {progress_callback}, status_callback: {status_callback}")
    
    process_task_manager.progress_callback = progress_callback
    process_task_manager.status_callback = status_callback
    
    process_task_manager.reset()
    
    print("[Process] 创建后台线程执行任务")
    thread = threading.Thread(target=run_process_task)
    thread.daemon = True
    thread.start()
    print("[Process] 后台线程已启动")


def stop_process_task():
    print("[Process] stop_process_task 被调用")
    process_task_manager.stop()


def get_process_status() -> Dict[str, Any]:
    return process_task_manager.get_status()