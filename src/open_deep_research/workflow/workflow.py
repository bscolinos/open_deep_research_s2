"""Simplified workflow implementation without LangGraph."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from open_deep_research.state import ReportState, SectionState
from open_deep_research.state_manager import SingleStoreStateManager
from open_deep_research.graph import (
    generate_report_plan,
    generate_queries,
    search_web,
    write_section,
    gather_completed_sections,
    write_final_sections,
    compile_final_report,
    initiate_final_section_writing,
)


async def run_workflow(
    topic: str,
    config: dict,
    manager: SingleStoreStateManager | None = None,
) -> ReportState:
    """Run the sequential workflow using SingleStoreDB for state."""

    manager = manager or SingleStoreStateManager()
    thread_id = config.get("thread_id", "default")

    state: ReportState = {
        "topic": topic,
        "messages": [],
        "sections": [],
        "completed_sections": [],
        "report_sections_from_research": "",
        "final_report": "",
        "source_str": "",
    }
    manager.save_report_state(thread_id, state)

    result = await generate_report_plan(state, {"configurable": config})
    state.update(result)
    manager.save_report_state(thread_id, state)

    for section in state["sections"]:
        sec_state: SectionState = {
            "topic": topic,
            "section": section,
            "search_iterations": 0,
            "search_queries": [],
            "source_str": "",
            "report_sections_from_research": "",
            "completed_sections": [],
            "messages": [],
        }
        q = await generate_queries(sec_state, {"configurable": config})
        sec_state.update(q)
        search_res = await search_web(sec_state, {"configurable": config})
        sec_state.update(search_res)
        res = await write_section(sec_state, {"configurable": config})
        sec_state.update(res)
        state["completed_sections"].extend(sec_state.get("completed_sections", []))
        manager.save_section_state(thread_id, section.name, sec_state)

    state.update(gather_completed_sections(state))
    manager.save_report_state(thread_id, state)

    final_sections = initiate_final_section_writing(state)
    for sec_state in final_sections:
        res = await write_final_sections(sec_state, {"configurable": config})
        sec_state.update(res)
        state["completed_sections"].extend(res["completed_sections"])
        manager.save_section_state(thread_id, sec_state["section"].name, sec_state)

    state.update(compile_final_report(state, {"configurable": config}))
    manager.save_report_state(thread_id, state)
    return state
