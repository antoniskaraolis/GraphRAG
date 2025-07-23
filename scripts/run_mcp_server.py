import sys, os

root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from server import mcp
from server.config import settings

import server.tools.graph
import server.tools.rag
import server.tools.clusters

if __name__ == "__main__":
    mcp.run(
        host=settings.HOST,
        port=settings.PORT,
        transport="http"
    )