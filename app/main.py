from fastapi import FastAPI
from app.api import routes_embeddings, routes_search, routes_prompts, routes_analysis

app = FastAPI(title="Prompt Similarity Service")

app.include_router(routes_embeddings.router, prefix="/api/embeddings")
app.include_router(routes_search.router, prefix="/api/search")
app.include_router(routes_prompts.router, prefix="/api/prompts")
app.include_router(routes_analysis.router, prefix="/api/analysis")