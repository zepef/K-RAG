from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class RefinementAnalysis(BaseModel):
    needs_refinement: bool = Field(
        description="Whether the user query needs refinement or clarification."
    )
    question: str = Field(
        description="A single clarifying question to ask the user (empty if no refinement needed)."
    )
    reasoning: str = Field(
        description="Brief explanation of why refinement is or isn't needed."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )
