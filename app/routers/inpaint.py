import asyncio
import io
import pickle
from typing import List
from loguru import logger
import redis.asyncio as redis
from fastapi import APIRouter, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.redis import conn
from app.dispatch.dispatcher import Dispatcher, DispatcherConfig
from app.container import dispatcher_list

inpaint_dispatcher_config = DispatcherConfig(
    worker_name="Inpainting",
    worker_stream_key="Inpainting_worker",
    worker_group_name="worker",
    ack_stream_key="Inpainting_finish_ack",
    ack_group_name="master"
)

inpaint_router = APIRouter()
inpaint_dispatcher = Dispatcher(conn, conn, conn, inpaint_dispatcher_config)
dispatcher_list.append(inpaint_dispatcher)


@inpaint_router.post('/upload')
async def upload_file(file: UploadFile):
    file_name = file.filename.split('.')[0]
    # file_name = file.filename
    await conn.set(file_name, pickle.dumps(file.file.read()))

    return {'status': 0, 'image_url': f'http://127.0.0.1:5003/image/{file.filename}'}


@inpaint_router.post('/inpaint')
async def inpaint_image(mask: UploadFile):
    # print(mask.filename)
    # with open(mask.filename,'wb') as f:
    #     f.write(mask.file.read())
    image_key = mask.filename.split('_')[0]
    # print(image_key)
    origin_pickle = await conn.get(image_key)

    result_url = await inpaint_dispatcher.dispatch({'image': origin_pickle, 'mask': pickle.dumps(mask.file.read())})

    return {'draw_url': f'http://127.0.0.1:5003{result_url}', "status": 1}


@inpaint_router.get('/tmp/ct/{file_name}')
async def get_ct_image(file_name: str):
    # file = open('samples/' + file_name, 'rb')
    # return StreamingResponse(file, media_type="image/jpeg")
    file_name = file_name.split('.')[0]
    img_byte = await conn.get(file_name)
    if img_byte:
        img_byte = pickle.loads(img_byte)
        img_file = io.BytesIO(img_byte)
        return StreamingResponse(img_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Item not found")


@inpaint_router.get('/image/{file_name}')
async def get_image(file_name: str):
    file_name = file_name.split('.')[0]
    img_byte = await conn.get(file_name)
    if img_byte:
        img_byte = pickle.loads(img_byte)
        img_file = io.BytesIO(img_byte)
        return StreamingResponse(img_file, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Item not found")
