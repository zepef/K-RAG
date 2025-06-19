# Neo4j Integration Guide for K-RAG Agent

This guide explains how to use the Neo4j knowledge graph integration with K-RAG Agent.

## Overview

The Neo4j integration is **supplementary** to the existing markdown-based project management system. It provides additional intelligence capabilities while maintaining all existing functionality:

### What Neo4j Adds (Supplementary Features)
- **Knowledge Graph**: Persistent storage of relationships between queries, searches, and results
- **Unused Path Tracking**: Tracking of unexplored search paths and refinement suggestions
- **Knowledge Gap Analysis**: Identification of topics mentioned but never fully explored  
- **Cross-Session Intelligence**: Smart recommendations based on historical patterns
- **Graph Visualization**: Visual exploration of research connections

### What Remains Unchanged
- ✅ **Markdown Project Storage**: All prompts and results continue to be saved as markdown files
- ✅ **Project Organization**: The `/projects` directory structure remains the same
- ✅ **Existing Workflows**: All current features work exactly as before
- ✅ **No Data Migration Required**: Neo4j is opt-in and doesn't affect existing projects

## Setup

### 1. Environment Variables

Add these to your `.env` file:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=krag2024
```

### 2. Start Services

Using Docker Compose:

```bash
docker-compose up -d
```

This starts:
- Neo4j (ports 7474 for browser, 7687 for bolt)
- Redis (port 6379)
- PostgreSQL (port 5433)
- LangGraph API (port 8123)

### 3. Initialize Neo4j

Run the initialization script:

```bash
cd backend
python src/agent/initialize_neo4j.py
```

### 4. Migrate Existing Projects (Optional)

If you have existing markdown projects:

```bash
python src/agent/migrate_to_neo4j.py
```

## API Endpoints

### Graph Visualization

- `GET /graph/query/{query_id}` - Get graph data for a specific query
- `GET /graph/project/{project_id}` - Get graph data for a project
- `GET /graph/statistics` - Get overall graph statistics

### Research Intelligence

- `POST /graph/recommendations` - Get research recommendations
- `GET /graph/similar-queries/{query_text}` - Find similar past queries
- `GET /graph/knowledge-gaps` - Get identified knowledge gaps

### Migration

- `POST /graph/migrate` - Trigger migration of markdown projects

## How It Works

### 1. Query Tracking

Every query submitted to K-RAG is tracked with:
- Original text and any refinements
- All generated search queries (executed and unexecuted)
- Web pages accessed and content extracted
- Final answer and citations
- Identified knowledge gaps

### 2. Intelligence Features

#### Similar Query Detection
- Uses embeddings to find semantically similar past queries
- Helps avoid duplicate research
- Surfaces relevant past findings

#### Unused Proposals
- Tracks search queries that were generated but not executed
- These become available for future similar queries
- Helps complete partially explored topics

#### Knowledge Gaps
- Identifies topics mentioned but never fully explored
- Prioritizes based on frequency of encounters
- Guides future research directions

### 3. Graph Structure

```
Project
  └── Query (with embeddings)
      ├── RefinementSuggestion
      ├── SearchQuery (executed/unexecuted)
      │   └── WebPage
      │       └── Content (with embeddings)
      └── Answer
          └── References → Content
```

## Neo4j Browser

Access the Neo4j browser at http://localhost:7474

Default credentials:
- Username: neo4j
- Password: krag2024

### Example Queries

Find all queries in a project:
```cypher
MATCH (q:Query)-[:BELONGS_TO]->(p:Project {name: "Your Project"})
RETURN q.text, q.timestamp
ORDER BY q.timestamp DESC
```

Find unused search proposals:
```cypher
MATCH (sq:SearchQuery {was_executed: false})
RETURN sq.query_text, sq.rationale
LIMIT 20
```

Visualize a query's full graph:
```cypher
MATCH (q:Query {id: "your-query-id"})
OPTIONAL MATCH (q)-[r*1..3]-(connected)
RETURN q, r, connected
```

## Performance Optimization

The system includes:
- Embedding caching to reduce computation
- Indexed fields for fast lookups
- Async operations throughout
- Connection pooling

## Security Considerations

- Sensitive content is hashed, not stored in full
- Project isolation ensures data separation
- Access control at the API level
- No credentials stored in graph

## Troubleshooting

### Connection Issues

If Neo4j connection fails:
1. Check if Neo4j is running: `docker ps`
2. Verify credentials in `.env`
3. Check logs: `docker logs krag-neo4j`

### Migration Issues

If migration fails:
1. Check markdown file format
2. Verify project directory structure
3. Review migration logs
4. Run with `--verify-only` flag first

### Performance Issues

If queries are slow:
1. Check if indices are created
2. Monitor Neo4j memory usage
3. Consider increasing heap size in docker-compose.yml

## Future Enhancements

Planned features include:
- Graph visualization UI component
- Collaborative filtering for team insights
- Automated research planning
- Export/import of knowledge graphs
- Integration with external knowledge bases