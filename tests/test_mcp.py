"""MCP server tests: start the server over stdio and call its tools like an agent would."""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def _call(env, calls):
    params = StdioServerParameters(command=sys.executable, args=["-m", "rag_optimizer.mcp_server"],
                                   env=env, cwd=str(ROOT))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = sorted(t.name for t in (await s.list_tools()).tools)
        out = [json.loads((await s.call_tool(n, a)).content[0].text) for n, a in calls]
        return names, out


def test_mcp_tools(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"run_id": "r0001", "config": "c128-bm25-k3-none", "dataset": "squad-v1.1-dev",
                               "split": "plan", "status": "ok",
                               "metrics": {"accuracy": 0.9, "p95_ms": 0.6, "tokens": 498.0}}) + "\n")
    tree = tmp_path / "tree.json"
    tree.write_text(json.dumps({"nodes": [{"config": "c128-bm25-k3-none", "status": "kept"}]}))
    extra = ROOT / "tests" / "fixtures" / "kb99_injected.md"
    env = {"RAG_OPT_LOG": str(log), "RAG_OPT_TREE": str(tree), "RAG_OPT_EXTRA_NOTES": str(extra),
           "PATH": "/usr/bin:/bin"}
    names, (research, exps, other_split, t) = asyncio.run(_call(env, [
        ("search_research", {"query": "BM25 versus dense retrieval for question answering"}),
        ("search_experiments", {"config_key": "c128-bm25-k3-none", "split": "plan"}),
        ("search_experiments", {"config_key": "", "split": "validation"}),
        ("get_tree", {}),
    ]))
    assert names == ["get_tree", "search_experiments", "search_research"]
    ids = [n["id"] for n in research["results"]]
    assert ids and "kb99" not in ids                      # injected note quarantined
    assert exps["runs"][0]["run_id"] == "r0001"
    assert other_split["runs"] == []                      # metadata filter by split
    assert t["nodes"][0]["status"] == "kept"
