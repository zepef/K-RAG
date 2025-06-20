import os
import asyncio

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    RefinementState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)
from agent.project_manager import ProjectManager
from agent.neo4j_manager import Neo4jManager
import uuid
import logging

logger = logging.getLogger(__name__)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Initialize Neo4j manager as a global instance
neo4j_manager = None

def get_neo4j_manager():
    """Get or create Neo4j manager instance."""
    global neo4j_manager
    if neo4j_manager is None:
        neo4j_manager = Neo4jManager()
    return neo4j_manager

# Nodes
async def refine_query(state: OverallState, config: RunnableConfig) -> RefinementState:
    """LangGraph node that evaluates user query and suggests refinements.
    
    Analyzes the user's initial query to determine if clarification is needed
    before proceeding with web research. If the query is ambiguous or could
    benefit from refinement, it suggests options for the user.
    
    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable
        
    Returns:
        Dictionary with refinement state including suggestions
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get the user's query from messages
    user_query = get_research_topic(state["messages"])
    
    # Store original query if not already stored
    if not state.get("original_query"):
        state["original_query"] = user_query
    
    # Check if user is ready to search (said "let's go" or similar)
    if state.get("user_ready_to_search", False):
        return {
            "needs_refinement": False,
            "refinement_suggestions": [],
            "clarified_query": user_query
        }
    
    # Check last message to see if user wants to proceed
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and last_message.type == "human":
        proceed_phrases = ["let's go", "lets go", "go ahead", "proceed", "search now", "that's enough", "start searching", "go", "ready"]
        if any(phrase in last_message.content.lower() for phrase in proceed_phrases):
            return {
                "needs_refinement": False,
                "refinement_suggestions": [],
                "clarified_query": user_query,
            }
    
    # Init LLM for refinement analysis
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.5,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    
    # Build conversation context
    conversation_context = ""
    if state.get("refinement_conversation"):
        conversation_context = "\n\nPrevious refinement conversation:\n"
        for msg in state["refinement_conversation"]:
            conversation_context += f"{msg['role']}: {msg['content']}\n"
    
    # Initialize session tracking if not present
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())
    if not state.get("query_id"):
        state["query_id"] = str(uuid.uuid4())
    
    # Create refinement prompt
    refinement_prompt = f"""Analyze the current state of the user's query and determine if further clarification would be helpful.

Original query: "{state.get('original_query', user_query)}"
Current understanding: "{user_query}"
{conversation_context}

Consider:
1. Based on the conversation so far, is there still ambiguity that needs clarification?
2. Would another clarifying question help narrow down the search intent?
3. Has the user provided enough context for effective research?

If the query is now sufficiently clear for research, respond with:
{{"needs_refinement": false, "question": "", "reasoning": "Query is sufficiently refined"}}

If further refinement would be helpful, provide ONE new clarifying question based on what's been discussed. Don't repeat previous questions.

Examples:
- After learning they want info about Python programming: {{"needs_refinement": true, "question": "Are you looking for beginner tutorials, advanced features, or something specific like web development with Python?", "reasoning": "The scope of Python programming is still broad"}}
- After clarifying Apple company: {{"needs_refinement": true, "question": "Are you interested in their products, financial performance, or recent news?", "reasoning": "Apple Inc. has many aspects to explore"}}

IMPORTANT: 
- Generate only ONE clarifying question
- Build on previous answers to dig deeper
- Always end your question by mentioning: '(or say "let's go" to start searching)'

Respond in JSON format."""

    # Get refinement analysis
    from agent.tools_and_schemas import RefinementAnalysis
    structured_llm = llm.with_structured_output(RefinementAnalysis)
    result = structured_llm.invoke(refinement_prompt)
    
    # Save refinement suggestions to Neo4j if any
    if result.needs_refinement and result.question:
        try:
            nm = get_neo4j_manager()
            await nm.save_query_session({
                "query": {
                    "id": state.get("query_id"),
                    "text": user_query,
                    "timestamp": datetime.now().isoformat(),
                    "was_refined": True
                },
                "refinement_suggestions": [{
                    "text": result.question,
                    "reasoning": result.reasoning,
                    "was_selected": False  # Will be updated if user selects it
                }],
                "project_id": state.get("project_id"),
                "session_id": state.get("session_id")
            })
        except Exception as e:
            logger.warning(f"Failed to save refinement to Neo4j: {e}")
    
    return {
        "needs_refinement": result.needs_refinement,
        "refinement_suggestions": [result.question] if result.needs_refinement and result.question else [],
        "clarified_query": user_query,
        "session_id": state.get("session_id"),
        "query_id": state.get("query_id")
    }


def should_refine(state: RefinementState) -> str:
    """Routing function to determine if refinement is needed."""
    if state["needs_refinement"] and not state.get("user_approved_refinement", False):
        return "wait_for_user"
    return "generate_query"


async def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    
    # Save search queries to Neo4j
    try:
        nm = get_neo4j_manager()
        search_queries = []
        for idx, sq in enumerate(result.query):
            search_queries.append({
                "id": f"{state.get('query_id')}_sq_{idx}",
                "query_text": sq.query,
                "rationale": sq.rationale,
                "execution_order": idx,
                "was_executed": idx < state.get("initial_search_query_count", 3)
            })
        
        # Get recommendations from Neo4j
        recommendations = await nm.get_research_recommendations(
            get_research_topic(state["messages"]),
            state.get("project_id")
        )
        
        # Log recommendations for visibility
        if recommendations["similar_queries"]:
            logger.info(f"Found {len(recommendations['similar_queries'])} similar past queries")
        if recommendations["unused_proposals"]:
            logger.info(f"Found {len(recommendations['unused_proposals'])} relevant unused proposals")
        
        # Save the generated queries
        await nm.save_query_session({
            "query": {
                "id": state.get("query_id"),
                "text": get_research_topic(state["messages"]),
                "timestamp": datetime.now().isoformat(),
                "was_refined": state.get("needs_refinement", False)
            },
            "search_queries": search_queries,
            "project_id": state.get("project_id"),
            "session_id": state.get("session_id")
        })
    except Exception as e:
        logger.warning(f"Failed to save search queries to Neo4j: {e}")
    
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


async def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    # Save web research results to Neo4j
    try:
        nm = get_neo4j_manager()
        webpage_data = {
            "id": f"{state.get('query_id', 'unknown')}_wp_{state.get('id', 0)}",
            "search_query_id": f"{state.get('query_id', 'unknown')}_sq_{state.get('id', 0)}",
            "url": resolved_urls[0] if resolved_urls else "unknown",
            "accessed_at": datetime.now().isoformat(),
            "contents": [{
                "id": f"{state.get('query_id', 'unknown')}_content_{state.get('id', 0)}_{idx}",
                "text": segment.get("text", ""),
                "type": "web_extract"
            } for idx, segment in enumerate(sources_gathered)]
        }
        
        # This is a partial save - we'll complete it in finalize_answer
        state["_neo4j_webpage_data"] = state.get("_neo4j_webpage_data", [])
        state["_neo4j_webpage_data"].append(webpage_data)
    except Exception as e:
        logger.warning(f"Failed to prepare web research for Neo4j: {e}")
    
    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


async def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    # Track knowledge gaps if identified
    knowledge_gaps = []
    if result.knowledge_gap:
        knowledge_gaps.append({"topic": result.knowledge_gap})
    
    # Store knowledge gaps for later saving
    state["_neo4j_knowledge_gaps"] = state.get("_neo4j_knowledge_gaps", [])
    state["_neo4j_knowledge_gaps"].extend(knowledge_gaps)
    
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


async def wait_for_user(state: OverallState) -> OverallState:
    """LangGraph node that waits for user input during refinement.
    
    This node sends a clarifying question to the user and waits for their response.
    The user can answer the question or proceed with the original query.
    
    Args:
        state: Current graph state containing refinement question
        
    Returns:
        Updated state with user's refinement choice
    """
    # Get the clarifying question
    question = state["refinement_suggestions"][0] if state.get("refinement_suggestions") else ""
    
    # Add the question to refinement conversation history
    refinement_conv = state.get("refinement_conversation", [])
    refinement_conv.append({"role": "assistant", "content": question})
    
    # Return state with AI message containing the clarifying question
    return {
        "messages": [AIMessage(content=question, additional_kwargs={"refinement_request": True})],
        "needs_refinement": True,
        "refinement_conversation": refinement_conv
    }


async def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    # Save prompt session if enabled and project name is provided (not default)
    saved_paths = state.get("saved_prompt_paths", [])
    project_name = state.get("project_name", "")
    
    # Only save if save_prompts is enabled, project_id exists, and project name is not empty/default
    if (state.get("save_prompts", False) and 
        state.get("project_id") and 
        project_name and 
        project_name.strip() != "" and 
        project_name != "Default Project"):
        
        # Extract data for saving
        original_query = state.get("original_query", get_research_topic(state["messages"]))
        
        # Handle search queries - they might be Query objects or strings
        raw_queries = state.get("search_query", [])
        search_queries = []
        for sq in raw_queries:
            if hasattr(sq, 'query') and hasattr(sq, 'rationale'):
                # It's a Query object
                search_queries.append({"query": sq.query, "rationale": sq.rationale})
            elif isinstance(sq, dict) and 'query' in sq:
                # It's already a dict
                search_queries.append(sq)
            elif isinstance(sq, str):
                # It's just a string
                search_queries.append({"query": sq, "rationale": "N/A"})
        
        research_results = state.get("web_research_result", [])
        sources = [source["value"] for source in unique_sources]
        
        # Save the session in a thread to avoid blocking
        def _save_session():
            project_manager = ProjectManager()
            return project_manager.save_prompt_session(
                project_id=state.get("project_id", "default"),
                project_name=state.get("project_name", "Default Project"),
                query=original_query,
                search_queries=search_queries,
                research_results=research_results,
                final_answer=result.content,
                sources=sources,
                metadata={
                    "reasoning_model": reasoning_model,
                    "research_loops": state.get("research_loop_count", 0),
                    "timestamp": current_date
                }
            )
        
        saved_path = await asyncio.to_thread(_save_session)
        saved_paths.append(saved_path)
        
        # Save complete session to Neo4j
        try:
            nm = get_neo4j_manager()
            
            # Prepare complete session data
            session_data = {
                "query": {
                    "id": state.get("query_id", str(uuid.uuid4())),
                    "text": original_query,
                    "timestamp": current_date,
                    "was_refined": state.get("needs_refinement", False)
                },
                "search_queries": search_queries,
                "webpages": state.get("_neo4j_webpage_data", []),
                "answer": {
                    "id": f"{state.get('query_id', 'unknown')}_answer",
                    "text": result.content,
                    "confidence_score": 1.0,
                    "citations": [{
                        "content_id": f"{state.get('query_id', 'unknown')}_content_0_{idx}",
                        "text": source.get("text", "")[:100],
                        "order_index": idx
                    } for idx, source in enumerate(unique_sources)]
                },
                "knowledge_gaps": state.get("_neo4j_knowledge_gaps", []),
                "project_id": state.get("project_id"),
                "project_name": project_name,
                "session_id": state.get("session_id")
            }
            
            await nm.save_query_session(session_data)
            logger.info(f"Saved session to Neo4j with query_id: {session_data['query']['id']}")
        except Exception as e:
            logger.error(f"Failed to save session to Neo4j: {e}")

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
        "saved_prompt_paths": saved_paths,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("refine_query", refine_query)
builder.add_node("wait_for_user", wait_for_user)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `refine_query`
# This means that this node is the first one called
builder.add_edge(START, "refine_query")
# Check if refinement is needed
builder.add_conditional_edges(
    "refine_query", should_refine, ["wait_for_user", "generate_query"]
)
# Wait for user input if refinement needed
builder.add_edge("wait_for_user", "refine_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="K-RAG Agent", interrupt_after=["wait_for_user"])
