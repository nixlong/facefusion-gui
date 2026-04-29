import sys
import time
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

START_TIME = time.time()

from .preview_api   import preview_router
from .process_api   import process_router
from .state_api     import state_router


def register_http_api(app: FastAPI) -> None:
    """注册 HTTP API 路由到 FastAPI 应用"""
    from facefusion import metadata

    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(preview_router)
    app.include_router(process_router)
    app.include_router(state_router)

    # 直接注册路由，避免使用 router
    @app.get("/api/v1/health")
    async def health_check():
        print("health_check")
        return {
            "success": True,
            "status": "healthy",
            "uptime": time.time() - START_TIME
        }

    @app.get("/api/v1/info")
    async def get_info():
        print("get_info")
        return {
            "success": True,
            "version": metadata.get("version"),
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }

    @app.post("/api/v1/shutdown")
    async def shutdown():
        import os
        os._exit(0)


def start_http_api_server() -> None:
    """启动独立的 HTTP API 服务器"""
    print("[Facefusion API] Starting HTTP API server...")
    try:
        from facefusion import metadata
        
        # 创建 FastAPI 应用
        api_app = FastAPI()
        
        # 注册 API 路由
        register_http_api(api_app)
        
        # 启动 HTTP API 服务器
        def run_api_server():
            print("[Facefusion API] HTTP API server starting on http://127.0.0.1:7863")
            uvicorn.run(api_app, host="127.0.0.1", port=7863)            
        
        # 在后台线程中启动 HTTP API 服务器
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        
        print("[Facefusion API] HTTP API server started successfully")
    except Exception as e:
        print(f"[Facefusion API] Failed to start HTTP API server: {e}")
        import traceback
        traceback.print_exc()
