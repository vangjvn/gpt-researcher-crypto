import json
import os
from typing import Dict, List
import time
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from backend.report_type import BasicReport, DetailedReport
from gpt_researcher.utils.enum import ReportType

from gpt_researcher.utils.enum import Dict_tone
from backend.server.websocket_manager import WebSocketManager
from multi_agents.main import run_research_task
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher.orchestrator.actions import stream_output
from backend.server.server_utils import (
    sanitize_filename, handle_start_command, handle_human_feedback,
    generate_report_files, send_file_paths, get_config_dict,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication, extract_command_data
)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ''
    GOOGLE_CX_KEY: str = ''
    BING_API_KEY: str = ''
    SEARCHAPI_API_KEY: str = ''
    SERPAPI_API_KEY: str = ''
    SERPER_API_KEY: str = ''
    SEARX_URL: str = ''


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)

# Routes


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/getConfig")
async def get_config(
    langchain_api_key: str = Header(None),
    openai_api_key: str = Header(None),
    tavily_api_key: str = Header(None),
    google_api_key: str = Header(None),
    google_cx_key: str = Header(None),
    bing_api_key: str = Header(None),
    searchapi_api_key: str = Header(None),
    serpapi_api_key: str = Header(None),
    serper_api_key: str = Header(None),
    searx_url: str = Header(None)
):
    return get_config_dict(
        langchain_api_key, openai_api_key, tavily_api_key,
        google_api_key, google_cx_key, bing_api_key,
        searchapi_api_key, serpapi_api_key, serper_api_key, searx_url
    )


@app.get("/files/")
async def list_files():
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/setConfig")
async def set_config(config: ConfigRequest):
    update_environment_variables(config.dict())
    return {"message": "Config updated successfully"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@app.post("/api/research")
async def research_endpoint(request: Request):
    try:
        # 准备参数
        data = await request.json()
        query = f"{data.get("task")}"
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{query}")
        tone = data.get("tone", "Objective").lower()
        tone = Dict_tone[tone]

        report_type = data.get("report_type", "research_report").lower()

        source_urls = ["https://cn.investing.com/crypto","https://mifengcha.com/","alternative.me/crypto/fear-and-greed-index/","coindesk.com","cointelegraph.com","reddit.com/r/CryptoCurrency"]

        # 默认research_report。1分钟左右，detailed_report：3分钟左右。multi_agents：5分钟左右。
        # 执行研究任务
        if report_type == "multi_agents": # multi_agents 任务 通过多个agent进行研究, 生成综合报告，返回报告内容和文件路径，耗时5分钟左右
            print("multi_agents")
            report = await run_research_task(
                query=query,
                tone=tone
            )
        elif report_type == "detailed_report": # detailed_report 任务 通过单个agent进行研究, 生成详细报告，耗时3分钟左右
            print("detailed_report")
            researcher = DetailedReport(
                query=query,
                report_type=ReportType.DetailedReport.value,
                report_source="web",
                source_urls=source_urls,
                tone=tone,
                config_path="",
                websocket=None,
                headers={}
            )
            report = await researcher.run()
        else: # basic_report 任务 通过单个agent进行研究, 生成基础报告，耗时30秒左右
            print("basic_report")
            researcher = BasicReport(
                query=query,
                report_type=report_type,
                report_source="web",
                source_urls=source_urls,
                tone=tone,
                config_path="",
                websocket=None,
                headers={}
            )
            report = await researcher.run()

        # 生成报告文件
        file_paths = await generate_report_files(str(report), sanitized_filename)
        new_file_paths = {}
        new_file_paths["docx"] = file_paths["docx"]
        new_file_paths["md"] = file_paths["md"]
        # 准备响应
        response = {
            "report": str(report),
            "file_paths": new_file_paths
        }

        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)