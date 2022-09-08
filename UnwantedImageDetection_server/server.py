from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import uvicorn
import config
import os
from utils import hashkey, base642array
from register import Register
from model import Model
import traceback


class Key(BaseModel):
    key: str


class Data(BaseModel):
    content: str
    user_id: Optional[int] = None
    threshold: Optional[float] = None


class Switch(BaseModel):
    model_name: str
    ctx_id: int
    key: str


model = Model(config.ENGINE_FILE_PATH, config.CLASS_NUM, config.CLASS_NAMES)

app = FastAPI()
register = Register(config.SERVER_IP, config.SERVER_PORT,
                    config.MANAGER_IP, config.MANAGER_PORT, 
                    config.MANAGER_INTERFACE_REGISTER, config.MANAGER_INTERFACE_CANCEL,
                    config.MODEL_TYPE, config.MODEL_VERSION)
register_test = Register(config.SERVER_IP, config.SERVER_PORT,
                         '172.27.3.131', 5000,
                         config.MANAGER_INTERFACE_REGISTER, config.MANAGER_INTERFACE_CANCEL,
                         config.MODEL_TYPE, config.MODEL_VERSION)
register_preview = Register(config.SERVER_IP, config.SERVER_PORT,
                            '172.27.3.131', 80,
                            config.MANAGER_INTERFACE_REGISTER, config.MANAGER_INTERFACE_CANCEL,
                            config.MODEL_TYPE, config.MODEL_VERSION)


@app.on_event('startup')
async def startup():
    if config.REGISTER:
        register()
        register_test()
        register_preview()


@app.on_event('shutdown')
async def shutdown():
    if config.REGISTER:
        await register.cancel()
        await register_test.cancel()
        await register_preview.cancel()


@app.post("/model/predict")
async def model_predict(data: Data):
    try:
        image = base642array(data.content)
    except:
        return {"code": 402, "data": []}
    try:
        data = model(image) 
    except:
        traceback.print_exc()
        return {"code": 500, "data": []}
    return {"code": 200, "data": data}


@app.get("/model/switch")
async def model_switch(switch: Switch):
    if switch.key != hashkey(config.KEY):
        return {"code": 400}
    if model.switch(switch.model_name, switch.ctx_id):
        return {"code": 200}
    return {"code": 400}


@app.get("/health")
async def health():
    if model.check_health():
        return {"code": 200}
    else:
        return {"code": 400}


@app.get("/status")
async def server_status(key: Key):
    if key.key != hashkey(config.KEY):
        return {"code": 400, "status": {}}
    return {"code": 200, "status": register.dump()}


@app.get("/model_stop")
async def model_stop(key: Key):
    if key.key != hashkey(config.KEY):
        return {"code": 400}
    print("stopped by manager server!")
    os._exit(1)


@app.get("/model_cancel")
async def model_cancel():
    if config.REGISTER:
        res = await register.cancel()
        if res:
            return {"code": 200}
    return {"code": 400}


if __name__ == "__main__":
    uvicorn.run(app='server:app', host=config.SERVER_IP, port=config.SERVER_PORT, workers=1, reload=False)
