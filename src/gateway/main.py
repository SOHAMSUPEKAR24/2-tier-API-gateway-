from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
import asyncio
from src.gateway.middleware import screen_request, escalation_queue
from src.gateway.tier2_worker import tier2_worker_loop

app = FastAPI(title="Adaptive Two-Tier Malware Detection Gateway")

@app.on_event("startup")
async def startup_event():
    # Start the background Tier 2 worker
    asyncio.create_task(tier2_worker_loop(escalation_queue))
    print("Gateway started. Tier 2 background worker initialized.")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str, background_tasks: BackgroundTasks):
    """
    Single ingress endpoint acting as the API Gateway reverse proxy entrypoint.
    All traffic routes through here to be screened by Algorithm 1.
    """
    return await screen_request(request, background_tasks)

if __name__ == "__main__":
    uvicorn.run("src.gateway.main:app", host="0.0.0.0", port=8000, reload=False)
