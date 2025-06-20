from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    needs_refinement: bool
    refinement_suggestions: list[str]
    user_approved_refinement: bool
    original_query: str
    refinement_conversation: Annotated[list, operator.add]
    user_ready_to_search: bool
    # Project management fields
    project_id: str
    project_name: str
    save_prompts: bool
    saved_prompt_paths: Annotated[list, operator.add]
    
    # Neo4j tracking fields
    session_id: str
    query_id: str
    _neo4j_webpage_data: list  # Internal tracking for web pages
    _neo4j_knowledge_gaps: list  # Internal tracking for knowledge gaps


class RefinementState(TypedDict):
    needs_refinement: bool
    refinement_suggestions: list[str]
    clarified_query: str
    
    
class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
