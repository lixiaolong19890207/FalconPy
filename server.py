import time

import uvicorn
from fastapi import FastAPI, Request
from api import api_router
from common.log import logger
from core import settings
from db import database

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_VERSION}/openapi.json"
)

app.include_router(api_router)


@app.on_event("startup")
async def startup() -> None:
    logger.info("start app link database")
    await database.connect()
    logger.info("success link database")


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("stop app disconnect database")
    await database.disconnect()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # X- 作为前缀代表专有自定义请求头
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
