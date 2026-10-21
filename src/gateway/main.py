from fastapi import FastAPI, Request, BackgroundTasks
import os
import uvicorn
import asyncio
from src.gateway.middleware import screen_request, escalation_queue
from src.gateway.tier2_worker import tier2_worker_loop

app = FastAPI(title="Adaptive Two-Tier Malware Detection Gateway")

@app.on_event("startup")
async def startup_event():
    # Pre-warm verifier so it is ready before traffic starts
    from src.gateway.middleware import get_verifier
    get_verifier()
    # Start the background Tier 2 workers (default 2 per uvicorn process for 8 total workers)
    n_workers = int(os.environ.get("GATEWAY_TIER2_WORKERS_PER_PROC", "2"))
    for _ in range(n_workers):
        asyncio.create_task(tier2_worker_loop(escalation_queue))
    from src.gateway.middleware import start_audit_worker
    start_audit_worker()
    print(f"Gateway started. Tier 2 background worker pool initialized ({n_workers} workers per proc).")

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/admin/mode/{mode}")
async def set_mode(mode: str):
    import src.gateway.middleware as mw
    if mode in ["two_tier", "pure_tier2", "no_inspection"]:
        mw.INSPECTION_MODE = mode
        mw.save_gateway_state({"mode": mode})
        return {"status": "ok", "mode": mw.INSPECTION_MODE}
    from fastapi import HTTPException
    raise HTTPException(status_code=400, detail="Invalid mode")

@app.post("/admin/reset")
async def reset_state():
    import time
    import src.gateway.middleware as mw
    while not mw.escalation_queue.empty():
        try:
            mw.escalation_queue.get_nowait()
            mw.escalation_queue.task_done()
        except Exception:
            break
    mw.inflight_events.clear()
    mw.tier2_verdicts.clear()
    mw.reset_telemetry_stats()
    mw.save_gateway_state({"reset_epoch": time.time()})
    return {"status": "reset_complete"}

@app.get("/admin/stats")
async def get_admin_stats():
    import src.gateway.middleware as mw
    return mw.get_telemetry_stats()

@app.post("/admin/escalation_config")
async def set_escalation_config(request: Request):
    import src.gateway.middleware as mw
    try:
        body = await request.json()
    except Exception:
        body = {}
    behavior = body.get("behavior")
    timeout_ms = body.get("timeout_ms")
    timeout_action = body.get("timeout_action")
    updates = {}
    if behavior:
        mw.ESCALATION_BEHAVIOR = behavior
        updates["behavior"] = behavior
    if timeout_ms is not None:
        mw.ESCALATION_TIMEOUT_MS = float(timeout_ms)
        updates["timeout_ms"] = float(timeout_ms)
    if timeout_action:
        mw.TIMEOUT_ACTION = timeout_action
        updates["timeout_action"] = timeout_action
    if updates:
        mw.save_gateway_state(updates)
    return {
        "behavior": mw.ESCALATION_BEHAVIOR,
        "timeout_ms": mw.ESCALATION_TIMEOUT_MS,
        "timeout_action": mw.TIMEOUT_ACTION,
    }



@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request, path: str, background_tasks: BackgroundTasks):
    """
    Single ingress endpoint acting as the API Gateway reverse proxy entrypoint.
    All traffic routes through here to be screened by Algorithm 1.
    """
    return await screen_request(request, background_tasks)

if __name__ == "__main__":
    uvicorn.run("src.gateway.main:app", host="0.0.0.0", port=8000, reload=False)

