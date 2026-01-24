from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import yaml
import os
import socket

app = FastAPI()

# 挂载静态文件目录（放置 index.html）
app.mount("/static", StaticFiles(directory="static"), name="static")

CONFIG_PATH = "region_config.yaml"
IMAGE_PATH = "camera_background.png"

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/camera_background.png")
def get_image():
    if os.path.exists(IMAGE_PATH):
        return FileResponse(IMAGE_PATH)
    return {"error": "Image not found"}

class SubRegionModel(BaseModel):
    sub_region_id: str
    name: str
    type: str
    polygon_vertices: List[List[int]]
    layer_index: Optional[int] = None
    edge_vector: Optional[List[float]] = None

class BaseRegionModel(BaseModel):
    region_id: str
    name: str
    arm_id: str
    polygon_vertices: List[List[int]]
    sub_regions: List[SubRegionModel] = []

@app.get("/config")
def get_config():
    if not os.path.exists(CONFIG_PATH):
        return {"base_regions": []}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        # 使用 UnsafeLoader 以支持 !!python/tuple 和 numpy 对象
        return yaml.load(f, Loader=yaml.UnsafeLoader)

@app.post("/save")
def save_config(data: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    
    # 获取本机IP
    host = "0.0.0.0"
    port = 8000
    
    print(f"Starting server at http://localhost:{port}")
    try:
        # 尝试获取局域网IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Network access: http://{local_ip}:{port}")
    except:
        pass
        
    uvicorn.run(app, host=host, port=port)