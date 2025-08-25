"""FastAPI service exposing AgentForge pipeline endpoints."""
from __future__ import annotations
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import logging

from providers_async import get_async_provider, shutdown_providers

logger = logging.getLogger("agentforge.api")

app = FastAPI(title="AgentForge API", version="0.1.0")


class PlanRequest(BaseModel):
    provider: str = Field(..., examples=["openai", "grok", "ollama"])
    use_case: str = Field(..., min_length=10)


class GenerateRequest(PlanRequest):
    plan: str = Field(..., min_length=20)


class PipelineRequest(PlanRequest):
    pass


class PlanResponse(BaseModel):
    plan: str


class CodeResponse(BaseModel):
    code: str


class TestResponse(BaseModel):
    tests: str


class PipelineResponse(BaseModel):
    plan: str
    code: str
    tests: str


async def _call(provider: str, prompt: str) -> str:
    client = get_async_provider(provider)
    return await client.complete(prompt)


@app.post("/plan", response_model=PlanResponse)
async def create_plan(req: PlanRequest):
    prompt = f"Design an AI agent for: {req.use_case}. Provide detailed architecture, tools, dependencies, and stepwise logic."
    try:
        plan = await _call(req.provider, prompt)
        return PlanResponse(plan=plan)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=CodeResponse)
async def generate_code(req: GenerateRequest):
    prompt = f"Generate production-grade Python agent code for this plan: {req.plan} Include modular design, error handling, logging, and tests placeholder."
    try:
        code = await _call(req.provider, prompt)
        return CodeResponse(code=code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline", response_model=PipelineResponse)
async def full_pipeline(req: PipelineRequest):
    plan_prompt = f"Design an AI agent for: {req.use_case}. Provide detailed architecture, tools, dependencies, and stepwise logic."
    code_prompt_tpl = "Generate production-grade Python agent code for this plan: {plan} Include modular design, error handling, logging, and tests placeholder."
    test_prompt_tpl = "Create pytest tests for this agent code: {code} Include edge cases."
    try:
        plan = await _call(req.provider, plan_prompt)
        code = await _call(req.provider, code_prompt_tpl.format(plan=plan))
        tests = await _call(req.provider, test_prompt_tpl.format(code=code))
        return PipelineResponse(plan=plan, code=code, tests=tests)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def _shutdown():
    await shutdown_providers()


@app.get("/health")
async def health():
    return {"status": "ok"}