"""API 路由集合"""
from fastapi import APIRouter

from . import chat, documents

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(chat.router)
