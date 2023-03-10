import io
import os
import pickle
import asyncio
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import fastapi
from starlette.middleware.cors import CORSMiddleware

from app.redis import conn
from app.container import dispatcher_list
import app.routers.inpaint as inpaint

app = fastapi.FastAPI()


@app.on_event('startup')
async def register_dispatcher_listener():
    for image_file in os.listdir('samples/'):
        # print(image_file)
        await conn.set(image_file.split('.')[0],pickle.dumps(open('samples/'+image_file,'rb').read()))

    for dispatcher in dispatcher_list:
        asyncio.create_task(dispatcher.wait_worker_response())

origins= [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5003",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
)

app.include_router(inpaint.inpaint_router)


@app.get('/result/{image_key}')
async def get_result(image_key: str):
    if image_key == 'error':
        return StreamingResponse(open('app/statics/failed.jpg','rb'),media_type='image/jpeg')
    img_byte = await conn.get(image_key)
    if img_byte:
        img_byte = pickle.loads(img_byte)
        img_file = io.BytesIO(img_byte)
        return StreamingResponse(img_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Item not found")
