from fastapi import APIRouter
from typing import Any, cast
from pydantic import BaseModel
from typing import List, Optional

from facefusion import state_manager
from facefusion.uis.http_server.models import ApiResponse, StateItemRequest, StateBatchRequest, FaceFusionStateItems

state_router = APIRouter()


@state_router.get("/api/v1/state_get", response_model=ApiResponse)
async def get_all_state():
    try:
        state = state_manager.get_state()
        return ApiResponse(success=True, data=dict(state))
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.get("/api/v1/state_get/{key}", response_model=ApiResponse)
async def get_state_item(key: str):
    try:
        value = state_manager.get_item(key)
        type_name = type(value).__name__
        return ApiResponse(success=True, data={key: value, "type": type_name})
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.post("/api/v1/state_set", response_model=ApiResponse)
async def set_state_item(params: FaceFusionStateItems):
    try:
        params_dict = params.dict(exclude_unset=True)
        for key, value in params_dict.items():
            state_manager.set_item(key, value)
            print(f'Update state item {key} to {value}')
        return ApiResponse(success=True, message="State updated")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))


@state_router.post("/api/v1/state_batch", response_model=ApiResponse)
async def set_batch_state(request: StateBatchRequest):
    try:
        batch_data = request.model_dump(exclude_none=True)
        for key, value in batch_data.items():
            state_manager.set_item(key, value)
        return ApiResponse(success=True, message="Batch state updated")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))

@state_router.post("/api/v1/state_clear/{key}", response_model=ApiResponse)
async def clear_state_item(key: str):
    try:
        value = state_manager.get_item(key)
        if value is not None :        
            state_manager.clear_item(key)
            print(f'Clear state item {key}')
        return ApiResponse(success=True, message="State cleared")
    except Exception as e:
        return ApiResponse(success=False, message=str(e))