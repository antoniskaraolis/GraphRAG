import asyncio
from fastmcp.client import Client
from server.config import settings

async def main():
    url = f"http://{settings.HOST}:{settings.PORT}/mcp"
    async with Client(url) as client:
        # Stats
        stats = await client.call_tool("graph/stats", {})
        print("graph/stats ->", stats.structured_content)

        # Search
        search_res = await client.call_tool(
            "graph/search", {"query": "machine learning for statistics", "top_k": 5}
        )
        results = search_res.structured_content["result"]
        print("graph/search ->", results)

        node_id = results[0]["id"]
        print("Using node_id:", node_id)

        # Neighbors
        neighs = await client.call_tool(
            "graph/neighbors", {"node_id": node_id, "relationship": None}
        )
        print("graph/neighbors ->", neighs.structured_content)

        # Clusters
        clist = await client.call_tool("clusters/list", {"n_clusters": 10})
        print("clusters/list ->", clist.structured_content)
        canal = await client.call_tool("clusters/analyze", {"n_clusters": 10})
        print("clusters/analyze ->", canal.structured_content)

        # RAG
        rag_res = await client.call_tool(
            "rag/query", {"question": "machine learning for statistics", "top_k": 2}
        )
        answer, ids = rag_res.data  
        print("rag/query ->", answer, ids)

        rag_ctx = await client.call_tool("rag/context", {"node_id": node_id})
        print("rag/context ->", rag_ctx.structured_content)

if __name__ == "__main__":
    asyncio.run(main())
