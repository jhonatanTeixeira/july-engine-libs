from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from .bridge_interface import BridgeInterface

bridge: BridgeInterface = None


def set_bridge(b: BridgeInterface):
    global bridge
    bridge = b


router = APIRouter(prefix="/search", tags=["Search"])


class WebSearchRequest(BaseModel):
    query: str
    search_depth: Optional[str] = "basic"
    include_answer: Optional[bool] = True
    max_results: Optional[int] = 5
    include_list: Optional[bool] = False
    model: Optional[str] = None
    describe_model: Optional[str] = None


class ScrapeUrlsRequest(BaseModel):
    urls: List[str]
    query: str
    context_window: Optional[int] = None
    model: Optional[str] = None


@router.post("/web", tags=["July"])
async def search_web(request: Request, payload: WebSearchRequest):
    headers = dict(request.headers)
    
    if "x-backend" not in headers:
        headers["x-backend"] = "api"
    
    result = await bridge.process_search_web(payload.model_dump(), headers)
    
    return {"result": result}


@router.post("/search-and-scrape", tags=["July"])
async def search_and_scrape(request: Request, payload: WebSearchRequest):
    headers = dict(request.headers)
    
    if "x-backend" not in headers:
        headers["x-backend"] = "api"
    
    # Ensure we get the list of results to scrape
    search_payload = payload.model_dump()
    search_payload["include_list"] = True
    
    results = await bridge.process_search_web(search_payload, headers)
    
    if not isinstance(results, list):
        return {"result": results}
        
    # Scrape and summarize (optional)
    result = await bridge.process_search_and_scrape(
        results, 
        payload.query, 
        headers, 
        describe_model=payload.describe_model
    )
    
    return {"result": result}


@router.post("/scrape-urls", tags=["July"])
async def scrape_urls(request: Request, payload: ScrapeUrlsRequest):
    """Raspa uma lista de URLs e sumariza via LLM com chunking por janela de contexto."""
    headers = dict(request.headers)
    result = await bridge.process_scrape_and_summarize(
        payload.urls,
        payload.query,
        headers,
        context_window=payload.context_window,
        model=payload.model,
    )
    return {"result": result}


@router.post("/github", tags=["July"])
async def search_github(request: Request):
    payload = await request.json()
    headers = dict(request.headers)
    
    if "x-backend" not in headers:
        headers["x-backend"] = "api"
    
    result = await bridge.process_search_code(payload, headers)
    
    return {"result": result}
