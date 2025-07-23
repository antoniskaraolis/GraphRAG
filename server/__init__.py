from fastmcp import FastMCP

tc = FastMCP(name="GraphRAG MCP Server")
global mcp
global __all__

mcp = tc
__all__ = ["mcp"]
