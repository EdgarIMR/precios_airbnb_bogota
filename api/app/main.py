"""API de BogoPrice: precio de referencia por noche para alojamientos en Bogotá."""
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

from app import config
from app.api import api_router

app = FastAPI(title=config.PROJECT_NAME, version=config.VERSION_API,
              openapi_url=f"{config.API_V1_STR}/openapi.json")

root_router = APIRouter()


@root_router.get("/")
def index() -> Any:
    return HTMLResponse("<html><body style='font-family:sans-serif;padding:16px'>"
                        "<h1>BogoPrice API</h1><p>Documentación: <a href='/docs'>/docs</a></p>"
                        "</body></html>")


app.include_router(api_router, prefix=config.API_V1_STR)
app.include_router(root_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
