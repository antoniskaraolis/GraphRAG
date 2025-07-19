# mcp_server/main.py
from fastmcp import FastMCP
from .endpoints import graph, rag, clusters
from .config import settings
import logging
import networkx as nx


#app = FastMCP(
    #title="GraphRAG MCP Server",
    #version="1.0.0",
    #description="Microservice Communication Protocol for GraphRAG system"
#)

app = FastMCP()

app.include_router(graph.router, prefix="/graph")
app.include_router(rag.router, prefix="/rag")
app.include_router(clusters.router, prefix="/clusters")

# logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("mcp-server")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting MCP Server")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"API Documentation: /docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
