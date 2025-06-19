"""Neo4j Manager for K-RAG Agent - Knowledge Graph Integration."""

from neo4j import AsyncGraphDatabase
from typing import List, Dict, Optional, Any
import asyncio
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages Neo4j graph database operations for K-RAG Agent."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize Neo4j connection and embedding model."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "krag2024")
        
        self.driver = AsyncGraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password),
            max_connection_lifetime=3600
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._embedding_cache = {}
    
    async def close(self):
        """Close Neo4j driver connection."""
        await self.driver.close()
    
    async def create_indices(self):
        """Create necessary indices for performance - Neo4j 2025 optimized."""
        queries = [
            # Text indices for exact matches (Block Format Optimized)
            "CREATE INDEX query_text_idx IF NOT EXISTS FOR (q:Query) ON (q.text)",
            "CREATE INDEX content_text_idx IF NOT EXISTS FOR (c:Content) ON (c.text)",
            "CREATE INDEX webpage_url_idx IF NOT EXISTS FOR (w:WebPage) ON (w.url)",
            "CREATE INDEX project_id_idx IF NOT EXISTS FOR (p:Project) ON (p.id)",
            
            # Composite indices for common queries
            "CREATE INDEX query_project_time_idx IF NOT EXISTS FOR (q:Query) ON (q.project_id, q.timestamp)",
            
            # Full-text search indices (Neo4j 2025 syntax)
            "CREATE FULLTEXT INDEX query_search_idx IF NOT EXISTS FOR (q:Query) ON EACH [q.text]",
            "CREATE FULLTEXT INDEX content_search_idx IF NOT EXISTS FOR (c:Content) ON EACH [c.text, c.summary]",
            
            # Unique constraints
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT query_id_unique IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT searchquery_id_unique IF NOT EXISTS FOR (sq:SearchQuery) REQUIRE sq.id IS UNIQUE",
            "CREATE CONSTRAINT webpage_id_unique IF NOT EXISTS FOR (w:WebPage) REQUIRE w.id IS UNIQUE",
            "CREATE CONSTRAINT content_id_unique IF NOT EXISTS FOR (c:Content) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT answer_id_unique IF NOT EXISTS FOR (a:Answer) REQUIRE a.id IS UNIQUE"
        ]
        
        # Vector indices with Neo4j 2025 optimizations
        vector_indices = [
            """CREATE VECTOR INDEX query_embedding_idx IF NOT EXISTS 
               FOR (q:Query) ON (q.embedding) 
               OPTIONS {indexConfig: {
                 `vector.dimensions`: 384,
                 `vector.similarity_function`: 'cosine',
                 `vector.quantization.enabled`: true,
                 `vector.quantization.type`: 'int8',
                 `vector.hnsw.m`: 16,
                 `vector.hnsw.ef_construction`: 200
               }}""",
            """CREATE VECTOR INDEX content_embedding_idx IF NOT EXISTS 
               FOR (c:Content) ON (c.embedding) 
               OPTIONS {indexConfig: {
                 `vector.dimensions`: 384,
                 `vector.similarity_function`: 'cosine',
                 `vector.quantization.enabled`: true,
                 `vector.quantization.type`: 'int8',
                 `vector.hnsw.m`: 16,
                 `vector.hnsw.ef_construction`: 200
               }}""",
            """CREATE VECTOR INDEX searchquery_embedding_idx IF NOT EXISTS 
               FOR (sq:SearchQuery) ON (sq.embedding) 
               OPTIONS {indexConfig: {
                 `vector.dimensions`: 384,
                 `vector.similarity_function`: 'cosine',
                 `vector.quantization.enabled`: true,
                 `vector.quantization.type`: 'int8',
                 `vector.hnsw.m`: 16,
                 `vector.hnsw.ef_construction`: 200
               }}"""
        ]
        
        async with self.driver.session() as session:
            # Create basic indices
            for query in queries:
                try:
                    await session.run(query)
                    logger.info(f"Created index: {query.split(' ')[2]}")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
            
            # Create vector indices (may fail on older Neo4j versions)
            for query in vector_indices:
                try:
                    await session.run(query)
                    logger.info("Created vector index")
                except Exception as e:
                    logger.warning(f"Vector index creation failed (requires Neo4j 5.11+): {e}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text with caching."""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Generate embedding
        embedding = self.embedder.encode(text).tolist()
        
        # Cache it
        self._embedding_cache[text_hash] = embedding
        
        # Limit cache size
        if len(self._embedding_cache) > 1000:
            # Remove oldest entries
            self._embedding_cache = dict(list(self._embedding_cache.items())[-500:])
        
        return embedding
    
    async def create_or_update_project(self, project_id: str, project_name: str) -> Dict[str, Any]:
        """Create or update a project node."""
        async with self.driver.session() as session:
            result = await session.run("""
                MERGE (p:Project {id: $project_id})
                ON CREATE SET 
                    p.name = $project_name,
                    p.created_at = datetime(),
                    p.session_count = 0
                ON MATCH SET 
                    p.name = $project_name,
                    p.last_updated = datetime(),
                    p.session_count = p.session_count + 1
                RETURN p
            """, project_id=project_id, project_name=project_name)
            
            record = await result.single()
            return dict(record["p"]) if record else None
    
    async def save_query_session(self, session_data: Dict[str, Any]) -> str:
        """Save complete query session to graph."""
        async with self.driver.session() as session:
            async with session.begin_transaction() as tx:
                # Create or update project if provided
                if session_data.get("project_id"):
                    await tx.run("""
                        MERGE (p:Project {id: $project_id})
                        ON CREATE SET p.name = $project_name, p.created_at = datetime()
                        ON MATCH SET p.last_updated = datetime()
                    """, 
                    project_id=session_data["project_id"],
                    project_name=session_data.get("project_name", "Default Project"))
                
                # Create Query node with embedding
                query_text = session_data["query"]["text"]
                query_embedding = await self.generate_embedding(query_text)
                
                query_result = await tx.run("""
                    CREATE (q:Query {
                        id: $id,
                        text: $text,
                        timestamp: datetime($timestamp),
                        project_id: $project_id,
                        session_id: $session_id,
                        was_refined: $was_refined,
                        embedding: $embedding
                    })
                    WITH q
                    MATCH (p:Project {id: $project_id})
                    CREATE (q)-[:BELONGS_TO]->(p)
                    RETURN q
                """, 
                id=session_data["query"]["id"],
                text=query_text,
                timestamp=session_data["query"].get("timestamp", datetime.now().isoformat()),
                project_id=session_data.get("project_id"),
                session_id=session_data.get("session_id"),
                was_refined=session_data["query"].get("was_refined", False),
                embedding=query_embedding)
                
                # Save refinement suggestions if any
                if session_data.get("refinement_suggestions"):
                    for idx, suggestion in enumerate(session_data["refinement_suggestions"]):
                        await tx.run("""
                            MATCH (q:Query {id: $query_id})
                            CREATE (rs:RefinementSuggestion {
                                id: $id,
                                text: $text,
                                reasoning: $reasoning,
                                order_index: $order_index,
                                was_selected: $was_selected,
                                timestamp: datetime()
                            })
                            CREATE (q)-[:RECEIVED_SUGGESTION]->(rs)
                        """,
                        query_id=session_data["query"]["id"],
                        id=f"{session_data['query']['id']}_ref_{idx}",
                        text=suggestion["text"],
                        reasoning=suggestion.get("reasoning", ""),
                        order_index=idx,
                        was_selected=suggestion.get("was_selected", False))
                
                # Create SearchQuery nodes
                for sq in session_data.get("search_queries", []):
                    sq_embedding = await self.generate_embedding(sq["query_text"])
                    await tx.run("""
                        MATCH (q:Query {id: $query_id})
                        CREATE (sq:SearchQuery {
                            id: $id,
                            query_text: $query_text,
                            rationale: $rationale,
                            was_executed: $was_executed,
                            execution_order: $execution_order,
                            embedding: $embedding,
                            timestamp: datetime()
                        })
                        CREATE (q)-[:GENERATED]->(sq)
                    """, 
                    query_id=session_data["query"]["id"],
                    id=sq["id"],
                    query_text=sq["query_text"],
                    rationale=sq.get("rationale", ""),
                    was_executed=sq.get("was_executed", True),
                    execution_order=sq.get("execution_order", 0),
                    embedding=sq_embedding)
                
                # Create WebPage and Content nodes
                for webpage_data in session_data.get("webpages", []):
                    # Create WebPage
                    await tx.run("""
                        MATCH (sq:SearchQuery {id: $search_query_id})
                        CREATE (wp:WebPage {
                            id: $id,
                            url: $url,
                            domain: $domain,
                            title: $title,
                            accessed_at: datetime($accessed_at)
                        })
                        CREATE (sq)-[:EXECUTED_SEARCH]->(wp)
                    """,
                    search_query_id=webpage_data["search_query_id"],
                    id=webpage_data["id"],
                    url=webpage_data["url"],
                    domain=webpage_data.get("domain", ""),
                    title=webpage_data.get("title", ""),
                    accessed_at=webpage_data.get("accessed_at", datetime.now().isoformat()))
                    
                    # Create Content nodes
                    for content in webpage_data.get("contents", []):
                        content_embedding = await self.generate_embedding(content["text"])
                        await tx.run("""
                            MATCH (wp:WebPage {id: $webpage_id})
                            CREATE (c:Content {
                                id: $id,
                                text: $text,
                                summary: $summary,
                                type: $type,
                                embedding: $embedding,
                                extracted_at: datetime()
                            })
                            CREATE (c)-[:FOUND_ON]->(wp)
                        """,
                        webpage_id=webpage_data["id"],
                        id=content["id"],
                        text=content["text"],
                        summary=content.get("summary", ""),
                        type=content.get("type", "general"),
                        embedding=content_embedding)
                
                # Create Answer node if provided
                if session_data.get("answer"):
                    answer_data = session_data["answer"]
                    answer_embedding = await self.generate_embedding(answer_data["text"])
                    
                    await tx.run("""
                        MATCH (q:Query {id: $query_id})
                        CREATE (a:Answer {
                            id: $id,
                            text: $text,
                            embedding: $embedding,
                            confidence_score: $confidence_score,
                            created_at: datetime()
                        })
                        CREATE (q)-[:RESULTED_IN]->(a)
                    """,
                    query_id=session_data["query"]["id"],
                    id=answer_data["id"],
                    text=answer_data["text"],
                    embedding=answer_embedding,
                    confidence_score=answer_data.get("confidence_score", 1.0))
                    
                    # Create citation relationships
                    for citation in answer_data.get("citations", []):
                        await tx.run("""
                            MATCH (a:Answer {id: $answer_id})
                            MATCH (c:Content {id: $content_id})
                            CREATE (a)-[:REFERENCES {
                                citation_text: $citation_text,
                                order_index: $order_index
                            }]->(c)
                        """,
                        answer_id=answer_data["id"],
                        content_id=citation["content_id"],
                        citation_text=citation.get("text", ""),
                        order_index=citation.get("order_index", 0))
                
                # Identify and create knowledge gaps
                for gap in session_data.get("knowledge_gaps", []):
                    await tx.run("""
                        MATCH (q:Query {id: $query_id})
                        MERGE (kg:KnowledgeGap {topic: $topic})
                        ON CREATE SET 
                            kg.id = randomUUID(),
                            kg.identified_at = datetime(),
                            kg.times_encountered = 1
                        ON MATCH SET 
                            kg.times_encountered = kg.times_encountered + 1,
                            kg.last_encountered = datetime()
                        CREATE (q)-[:IDENTIFIED_GAP]->(kg)
                    """,
                    query_id=session_data["query"]["id"],
                    topic=gap["topic"])
                
                await tx.commit()
                return session_data["query"]["id"]
    
    async def find_similar_queries(self, query_text: str, threshold: float = 0.8, limit: int = 10) -> List[Dict[str, Any]]:
        """Find queries similar to the current one using embeddings."""
        query_embedding = await self.generate_embedding(query_text)
        
        async with self.driver.session() as session:
            # Note: Neo4j doesn't have built-in cosine similarity for arrays
            # We'll fetch candidates and compute similarity in Python
            result = await session.run("""
                MATCH (q:Query)
                WHERE q.embedding IS NOT NULL
                RETURN q.id as id, q.text as text, q.embedding as embedding, 
                       q.timestamp as timestamp, q.project_id as project_id
                ORDER BY q.timestamp DESC
                LIMIT 100
            """)
            
            similar_queries = []
            async for record in result:
                # Compute cosine similarity
                embedding = record["embedding"]
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                if similarity >= threshold:
                    similar_queries.append({
                        "id": record["id"],
                        "text": record["text"],
                        "similarity": similarity,
                        "timestamp": record["timestamp"],
                        "project_id": record["project_id"]
                    })
            
            # Sort by similarity and limit
            similar_queries.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_queries[:limit]
    
    async def get_unused_proposals(self, query_text: str, threshold: float = 0.7, limit: int = 5) -> List[Dict[str, Any]]:
        """Get unused search proposals relevant to current query."""
        query_embedding = await self.generate_embedding(query_text)
        
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (q:Query)-[:GENERATED]->(sq:SearchQuery)
                WHERE sq.was_executed = false AND sq.embedding IS NOT NULL
                RETURN sq.id as id, sq.query_text as query_text, sq.rationale as rationale,
                       sq.embedding as embedding, q.timestamp as timestamp
                ORDER BY q.timestamp DESC
                LIMIT 100
            """)
            
            relevant_proposals = []
            async for record in result:
                similarity = self._cosine_similarity(query_embedding, record["embedding"])
                
                if similarity >= threshold:
                    relevant_proposals.append({
                        "id": record["id"],
                        "query_text": record["query_text"],
                        "rationale": record["rationale"],
                        "similarity": similarity,
                        "timestamp": record["timestamp"]
                    })
            
            relevant_proposals.sort(key=lambda x: x["similarity"], reverse=True)
            return relevant_proposals[:limit]
    
    async def identify_knowledge_gaps(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Identify topics that appear frequently but are never fully explored."""
        query = """
        MATCH (q:Query)-[:GENERATED]->(sq:SearchQuery)
        WHERE sq.was_executed = false
        WITH sq.query_text as unexplored_topic, COUNT(*) as frequency
        WHERE frequency > 2
        
        OPTIONAL MATCH (q2:Query)-[:GENERATED]->(sq2:SearchQuery)
        WHERE sq2.was_executed = true 
        AND sq2.query_text CONTAINS unexplored_topic
        
        WITH unexplored_topic, frequency, COUNT(sq2) as explored_count
        WHERE explored_count = 0
        
        MERGE (kg:KnowledgeGap {topic: unexplored_topic})
        ON CREATE SET 
            kg.id = randomUUID(),
            kg.identified_at = datetime(),
            kg.times_encountered = frequency,
            kg.priority_score = frequency * 1.5
        ON MATCH SET 
            kg.times_encountered = kg.times_encountered + frequency,
            kg.last_encountered = datetime(),
            kg.priority_score = kg.times_encountered * 1.5
        
        RETURN kg
        ORDER BY kg.priority_score DESC
        """
        
        params = {}
        if project_id:
            query = query.replace("MATCH (q:Query)", "MATCH (q:Query {project_id: $project_id})")
            params["project_id"] = project_id
        
        async with self.driver.session() as session:
            result = await session.run(query, **params)
            gaps = []
            async for record in result:
                gaps.append(dict(record["kg"]))
            return gaps
    
    async def get_research_recommendations(self, query_text: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive research recommendations based on historical patterns."""
        recommendations = {
            "similar_queries": await self.find_similar_queries(query_text),
            "unused_proposals": await self.get_unused_proposals(query_text),
            "knowledge_gaps": await self.identify_knowledge_gaps(project_id),
            "suggested_paths": [],
            "related_projects": []
        }
        
        # Find successful research paths from similar queries
        if recommendations["similar_queries"]:
            similar_query_ids = [q["id"] for q in recommendations["similar_queries"][:3]]
            
            async with self.driver.session() as session:
                # Get successful search patterns
                result = await session.run("""
                    MATCH (q:Query)-[:GENERATED]->(sq:SearchQuery)
                    WHERE q.id IN $query_ids AND sq.was_executed = true
                    WITH sq.query_text as search_pattern, COUNT(*) as usage_count
                    RETURN search_pattern, usage_count
                    ORDER BY usage_count DESC
                    LIMIT 5
                """, query_ids=similar_query_ids)
                
                paths = []
                async for record in result:
                    paths.append({
                        "search_pattern": record["search_pattern"],
                        "usage_count": record["usage_count"],
                        "confidence": record["usage_count"] / len(similar_query_ids)
                    })
                recommendations["suggested_paths"] = paths
                
                # Find related projects
                result = await session.run("""
                    MATCH (q:Query)-[:BELONGS_TO]->(p:Project)
                    WHERE q.id IN $query_ids AND p.id <> $current_project_id
                    RETURN DISTINCT p.id as id, p.name as name, COUNT(q) as relevance_count
                    ORDER BY relevance_count DESC
                    LIMIT 3
                """, query_ids=similar_query_ids, current_project_id=project_id or "")
                
                projects = []
                async for record in result:
                    projects.append({
                        "id": record["id"],
                        "name": record["name"],
                        "relevance_count": record["relevance_count"]
                    })
                recommendations["related_projects"] = projects
        
        return recommendations
    
    async def get_graph_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        base_match = "MATCH (p:Project {id: $project_id})" if project_id else ""
        project_filter = "WHERE q.project_id = $project_id" if project_id else ""
        
        async with self.driver.session() as session:
            # Node counts
            stats = {}
            
            # Total queries
            result = await session.run(f"""
                MATCH (q:Query) {project_filter}
                RETURN COUNT(q) as count
            """, project_id=project_id)
            record = await result.single()
            stats["total_queries"] = record["count"] if record else 0
            
            # Executed vs proposed searches
            result = await session.run(f"""
                MATCH (sq:SearchQuery)
                {"WHERE EXISTS ((q:Query {project_id: $project_id})-[:GENERATED]->(sq))" if project_id else ""}
                RETURN sq.was_executed as executed, COUNT(sq) as count
            """, project_id=project_id)
            
            stats["search_queries"] = {"executed": 0, "proposed": 0}
            async for record in result:
                if record["executed"]:
                    stats["search_queries"]["executed"] = record["count"]
                else:
                    stats["search_queries"]["proposed"] = record["count"]
            
            # Content and sources
            result = await session.run("""
                MATCH (c:Content)
                RETURN COUNT(c) as content_count,
                       COUNT(DISTINCT (c)-[:FOUND_ON]->(:WebPage)) as webpage_count
            """)
            record = await result.single()
            if record:
                stats["total_content"] = record["content_count"]
                stats["total_webpages"] = record["webpage_count"]
            
            # Knowledge gaps
            result = await session.run("""
                MATCH (kg:KnowledgeGap)
                RETURN COUNT(kg) as count, AVG(kg.priority_score) as avg_priority
            """)
            record = await result.single()
            if record:
                stats["knowledge_gaps"] = {
                    "count": record["count"],
                    "avg_priority": record["avg_priority"] or 0
                }
            
            return stats
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def migrate_markdown_projects(self, projects_dir: Path) -> Dict[str, Any]:
        """Import existing markdown files into the graph database."""
        imported_count = 0
        failed_count = 0
        
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                # Extract project info from directory name
                parts = project_dir.name.rsplit('_', 1)
                if len(parts) == 2:
                    project_name = parts[0].replace('_', ' ')
                    project_id = parts[1]
                    
                    # Create project node
                    await self.create_or_update_project(project_id, project_name)
                    
                    # Import each markdown session
                    for md_file in project_dir.glob('*.md'):
                        if md_file.name != 'index.md':
                            try:
                                session_data = await self._parse_markdown_session(md_file, project_id, project_name)
                                if session_data:
                                    await self.save_query_session(session_data)
                                    imported_count += 1
                            except Exception as e:
                                logger.error(f"Failed to import {md_file}: {e}")
                                failed_count += 1
        
        return {
            "imported": imported_count,
            "failed": failed_count,
            "status": "completed"
        }
    
    async def _parse_markdown_session(self, md_file: Path, project_id: str, project_name: str) -> Optional[Dict[str, Any]]:
        """Parse a markdown file to extract session data."""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract timestamp from filename
            parts = md_file.stem.split('_', 2)
            if len(parts) >= 3:
                date_str = parts[0]
                time_str = parts[1].replace('-', ':')
                timestamp = f"{date_str}T{time_str}"
            else:
                timestamp = datetime.now().isoformat()
            
            # Parse content sections
            lines = content.split('\n')
            
            # Extract query
            query_text = ""
            for i, line in enumerate(lines):
                if line.strip().startswith("> ") and i > 0 and "Original Query" in lines[i-2]:
                    query_text = line.strip()[2:]
                    break
            
            if not query_text:
                return None
            
            # Create session data structure
            session_data = {
                "project_id": project_id,
                "project_name": project_name,
                "session_id": md_file.stem,
                "query": {
                    "id": f"{project_id}_{md_file.stem}",
                    "text": query_text,
                    "timestamp": timestamp,
                    "was_refined": False
                },
                "search_queries": [],
                "webpages": [],
                "answer": None,
                "knowledge_gaps": []
            }
            
            # Extract search queries
            in_search_section = False
            query_count = 0
            for i, line in enumerate(lines):
                if "## Generated Search Queries" in line:
                    in_search_section = True
                elif in_search_section and line.startswith("## "):
                    in_search_section = False
                elif in_search_section and line.startswith("**Query:**"):
                    query_text = line.replace("**Query:**", "").strip()
                    # Look for rationale in next lines
                    rationale = ""
                    if i + 1 < len(lines) and "**Rationale:**" in lines[i + 1]:
                        rationale = lines[i + 1].replace("**Rationale:**", "").strip()
                    
                    session_data["search_queries"].append({
                        "id": f"{session_data['query']['id']}_sq_{query_count}",
                        "query_text": query_text,
                        "rationale": rationale,
                        "was_executed": True,
                        "execution_order": query_count
                    })
                    query_count += 1
            
            # Extract final answer
            in_answer_section = False
            answer_lines = []
            for line in lines:
                if "## Final Answer" in line:
                    in_answer_section = True
                elif in_answer_section and line.startswith("## "):
                    break
                elif in_answer_section and line.strip():
                    answer_lines.append(line)
            
            if answer_lines:
                session_data["answer"] = {
                    "id": f"{session_data['query']['id']}_answer",
                    "text": "\n".join(answer_lines),
                    "confidence_score": 1.0,
                    "citations": []
                }
            
            return session_data
            
        except Exception as e:
            logger.error(f"Error parsing markdown file {md_file}: {e}")
            return None