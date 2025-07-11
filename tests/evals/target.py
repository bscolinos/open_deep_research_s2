from typing import Literal
import uuid

from langchain_core.messages import MessageLikeRepresentation
from open_deep_research.graph import run_graph
from open_deep_research.multi_agent import run_multi_agent
from open_deep_research.state_manager import SingleStoreStateManager


async def generate_report_workflow(
    query: str,
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None,
    include_source: bool = True
):
    """Generate a report using the open deep research workflow"""
    manager = SingleStoreStateManager()
    config = {
        "thread_id": str(uuid.uuid4()),
    }
    if include_source:
        config["include_source_str"] = True
    if process_search_results:
        config["process_search_results"] = process_search_results
    return await run_graph(query, config, manager)


async def generate_report_multi_agent(
    messages: list[MessageLikeRepresentation],
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None,
    include_source: bool = True
):
    """Generate a report using the open deep research multi-agent architecture"""
    manager = SingleStoreStateManager()
    config = {}
    if include_source:
        config["include_source_str"] = True
    if process_search_results:
        config["process_search_results"] = process_search_results
    return await run_multi_agent(messages, config, manager)