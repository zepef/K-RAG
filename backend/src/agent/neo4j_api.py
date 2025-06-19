"""API endpoints for Neo4j graph operations."""

from fastapi import APIRouter, HTTPException, Query as QueryParam
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import asyncio
from datetime import datetime

from agent.neo4j_manager import Neo4jManager

router = APIRouter(prefix="/graph", tags=["graph"])

# Response models
class GraphNode(BaseModel):
    """Graph node representation."""
    id: str
    label: str
    properties: Dict[str, Any]

class GraphEdge(BaseModel):
    """Graph edge representation."""
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}

class GraphData(BaseModel):
    """Complete graph data structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class SimilarQuery(BaseModel):
    """Similar query result."""
    id: str
    text: str
    similarity: float
    timestamp: str
    project_id: Optional[str]

class UnusedProposal(BaseModel):
    """Unused search proposal."""
    id: str
    query_text: str
    rationale: str
    similarity: float
    timestamp: str

class KnowledgeGap(BaseModel):
    """Knowledge gap information."""
    id: str
    topic: str
    times_encountered: int
    priority_score: float
    last_encountered: Optional[str]

class ResearchRecommendations(BaseModel):
    """Complete research recommendations."""
    similar_queries: List[SimilarQuery]
    unused_proposals: List[UnusedProposal]
    knowledge_gaps: List[KnowledgeGap]
    suggested_paths: List[Dict[str, Any]]
    related_projects: List[Dict[str, Any]]

class GraphStatistics(BaseModel):
    """Graph statistics."""
    total_queries: int
    search_queries: Dict[str, int]
    total_content: int
    total_webpages: int
    knowledge_gaps: Dict[str, Any]


@router.get("/query/{query_id}", response_model=GraphData)
async def get_query_graph(query_id: str):
    """Get the graph data for a specific query and its related nodes."""
    nm = Neo4jManager()
    
    try:
        async with nm.driver.session() as session:
            # Get query and all related nodes
            result = await session.run("""
                MATCH (q:Query {id: $query_id})
                OPTIONAL MATCH (q)-[r1:BELONGS_TO]->(p:Project)
                OPTIONAL MATCH (q)-[r2:GENERATED]->(sq:SearchQuery)
                OPTIONAL MATCH (sq)-[r3:EXECUTED_SEARCH]->(wp:WebPage)
                OPTIONAL MATCH (c:Content)-[r4:FOUND_ON]->(wp)
                OPTIONAL MATCH (q)-[r5:RESULTED_IN]->(a:Answer)
                OPTIONAL MATCH (a)-[r6:REFERENCES]->(c2:Content)
                
                WITH q, p, sq, wp, c, a, c2,
                     COLLECT(DISTINCT r1) + COLLECT(DISTINCT r2) + 
                     COLLECT(DISTINCT r3) + COLLECT(DISTINCT r4) + 
                     COLLECT(DISTINCT r5) + COLLECT(DISTINCT r6) as relationships
                
                RETURN 
                    COLLECT(DISTINCT q) + COLLECT(DISTINCT p) + 
                    COLLECT(DISTINCT sq) + COLLECT(DISTINCT wp) + 
                    COLLECT(DISTINCT c) + COLLECT(DISTINCT a) + 
                    COLLECT(DISTINCT c2) as nodes,
                    relationships
            """, query_id=query_id)
            
            record = await result.single()
            if not record:
                raise HTTPException(status_code=404, detail="Query not found")
            
            nodes = []
            edges = []
            
            # Process nodes
            for node in record["nodes"]:
                if node is not None:
                    node_dict = dict(node)
                    labels = list(node.labels)
                    
                    # Remove embedding from properties for API response
                    if 'embedding' in node_dict:
                        del node_dict['embedding']
                    
                    nodes.append(GraphNode(
                        id=node_dict.get('id', str(node.id)),
                        label=labels[0] if labels else 'Unknown',
                        properties=node_dict
                    ))
            
            # Process relationships
            for rel in record["relationships"]:
                if rel is not None:
                    edges.append(GraphEdge(
                        source=str(rel.start_node.get('id', rel.start_node.id)),
                        target=str(rel.end_node.get('id', rel.end_node.id)),
                        type=rel.type,
                        properties=dict(rel)
                    ))
            
            return GraphData(nodes=nodes, edges=edges)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.get("/project/{project_id}", response_model=GraphData)
async def get_project_graph(
    project_id: str,
    limit: int = QueryParam(100, description="Maximum number of queries to include")
):
    """Get the graph data for an entire project."""
    nm = Neo4jManager()
    
    try:
        async with nm.driver.session() as session:
            # Get project and recent queries
            result = await session.run("""
                MATCH (p:Project {id: $project_id})
                OPTIONAL MATCH (q:Query)-[:BELONGS_TO]->(p)
                WITH p, q ORDER BY q.timestamp DESC LIMIT $limit
                
                OPTIONAL MATCH (q)-[r:GENERATED]->(sq:SearchQuery)
                OPTIONAL MATCH (q)-[r2:RESULTED_IN]->(a:Answer)
                
                WITH p, COLLECT(DISTINCT q) as queries, 
                     COLLECT(DISTINCT sq) as search_queries,
                     COLLECT(DISTINCT a) as answers,
                     COLLECT(DISTINCT r) + COLLECT(DISTINCT r2) as relationships
                
                RETURN p, queries, search_queries, answers, relationships
            """, project_id=project_id, limit=limit)
            
            record = await result.single()
            if not record or not record["p"]:
                raise HTTPException(status_code=404, detail="Project not found")
            
            nodes = []
            edges = []
            
            # Add project node
            project = dict(record["p"])
            nodes.append(GraphNode(
                id=project['id'],
                label='Project',
                properties=project
            ))
            
            # Add query nodes
            for q in record["queries"]:
                if q:
                    query_dict = dict(q)
                    if 'embedding' in query_dict:
                        del query_dict['embedding']
                    nodes.append(GraphNode(
                        id=query_dict['id'],
                        label='Query',
                        properties=query_dict
                    ))
                    
                    # Add edge from query to project
                    edges.append(GraphEdge(
                        source=query_dict['id'],
                        target=project_id,
                        type='BELONGS_TO'
                    ))
            
            # Add search query nodes and other entities
            for sq in record["search_queries"]:
                if sq:
                    sq_dict = dict(sq)
                    if 'embedding' in sq_dict:
                        del sq_dict['embedding']
                    nodes.append(GraphNode(
                        id=sq_dict['id'],
                        label='SearchQuery',
                        properties=sq_dict
                    ))
            
            # Add answer nodes
            for a in record["answers"]:
                if a:
                    answer_dict = dict(a)
                    if 'embedding' in answer_dict:
                        del answer_dict['embedding']
                    nodes.append(GraphNode(
                        id=answer_dict['id'],
                        label='Answer',
                        properties=answer_dict
                    ))
            
            # Add relationships
            for rel in record["relationships"]:
                if rel:
                    edges.append(GraphEdge(
                        source=str(rel.start_node.get('id')),
                        target=str(rel.end_node.get('id')),
                        type=rel.type,
                        properties=dict(rel)
                    ))
            
            return GraphData(nodes=nodes, edges=edges)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.post("/recommendations", response_model=ResearchRecommendations)
async def get_research_recommendations(
    query_text: str,
    project_id: Optional[str] = None
):
    """Get research recommendations based on query text."""
    nm = Neo4jManager()
    
    try:
        recommendations = await nm.get_research_recommendations(query_text, project_id)
        
        return ResearchRecommendations(
            similar_queries=[
                SimilarQuery(**q) for q in recommendations["similar_queries"]
            ],
            unused_proposals=[
                UnusedProposal(**p) for p in recommendations["unused_proposals"]
            ],
            knowledge_gaps=[
                KnowledgeGap(**g) for g in recommendations["knowledge_gaps"]
            ],
            suggested_paths=recommendations["suggested_paths"],
            related_projects=recommendations["related_projects"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.get("/statistics", response_model=GraphStatistics)
async def get_graph_statistics(project_id: Optional[str] = None):
    """Get overall graph statistics."""
    nm = Neo4jManager()
    
    try:
        stats = await nm.get_graph_statistics(project_id)
        
        return GraphStatistics(
            total_queries=stats.get("total_queries", 0),
            search_queries=stats.get("search_queries", {}),
            total_content=stats.get("total_content", 0),
            total_webpages=stats.get("total_webpages", 0),
            knowledge_gaps=stats.get("knowledge_gaps", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.post("/migrate")
async def migrate_projects():
    """Trigger migration of existing markdown projects to Neo4j."""
    from pathlib import Path
    
    nm = Neo4jManager()
    
    try:
        # Create indices first
        await nm.create_indices()
        
        # Run migration
        projects_dir = Path("/mnt/e/Projects/K-RAG/projects")
        results = await nm.migrate_markdown_projects(projects_dir)
        
        return {
            "status": "completed",
            "imported": results["imported"],
            "failed": results["failed"],
            "message": f"Successfully imported {results['imported']} sessions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.get("/knowledge-gaps", response_model=List[KnowledgeGap])
async def get_knowledge_gaps(
    project_id: Optional[str] = None,
    limit: int = QueryParam(10, description="Maximum number of gaps to return")
):
    """Get identified knowledge gaps."""
    nm = Neo4jManager()
    
    try:
        gaps = await nm.identify_knowledge_gaps(project_id)
        
        # Sort by priority and limit
        gaps.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        gaps = gaps[:limit]
        
        return [
            KnowledgeGap(
                id=g.get("id", ""),
                topic=g.get("topic", ""),
                times_encountered=g.get("times_encountered", 0),
                priority_score=g.get("priority_score", 0),
                last_encountered=g.get("last_encountered", None)
            )
            for g in gaps
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()


@router.get("/similar-queries/{query_text}", response_model=List[SimilarQuery])
async def find_similar_queries(
    query_text: str,
    threshold: float = QueryParam(0.8, description="Similarity threshold (0-1)"),
    limit: int = QueryParam(10, description="Maximum number of results")
):
    """Find queries similar to the provided text."""
    nm = Neo4jManager()
    
    try:
        similar = await nm.find_similar_queries(query_text, threshold, limit)
        
        return [
            SimilarQuery(
                id=q["id"],
                text=q["text"],
                similarity=q["similarity"],
                timestamp=q["timestamp"],
                project_id=q.get("project_id")
            )
            for q in similar
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await nm.close()