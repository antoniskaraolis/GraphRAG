from fastmcp import FastMCP

# Single shared MCP server instance
tc = FastMCP(name="GraphRAG MCP Server")
# alias as mcp for consistency
global mcp
global __all__

mcp = tc
__all__ = ["mcp"]