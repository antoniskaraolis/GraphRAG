#!/usr/bin/env python3
# scripts/mcp_server.py
from mcp_server.main import app

if __name__ == "__main__":
    import uvicorn
    from mcp_server.config import settings
    
    uvicorn.run(
        "mcp_server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENV == "development",
        log_level=settings.LOG_LEVEL.lower()
    )
