# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

K-RAG Agent is a research assistant application that uses LangGraph to create an AI agent capable of performing iterative web research with reflection. It consists of a Python backend (LangGraph/FastAPI) and a React/TypeScript frontend.

## Project Management

Research sessions can be saved to markdown files organized by project:
- Projects are stored in `/projects/` at the root of the repository
- Each project has its own folder named `ProjectName_projXXXXX`
- Sessions are saved as timestamped markdown files within project folders
- Leave project name empty for one-time queries that won't be saved

## Development Commands

### Quick Start
```bash
# Run both frontend and backend development servers
make dev

# Run individual services
make dev-frontend   # Frontend only (port 5173)
make dev-backend    # Backend only (port 8000)
```

### Backend Development (from /backend)
```bash
# Install dependencies
pip install .

# Run development server
langgraph dev

# Run tests
uv run pytest

# Linting and formatting
make lint     # Run ruff and mypy
make format   # Auto-format code

# Test the agent via CLI
python examples/cli_research.py "your research question"
```

### Frontend Development (from /frontend)
```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build

# Linting
npm run lint

# Preview production build
npm run preview
```

## Architecture Overview

### Backend Agent Graph
The agent follows a state machine pattern with these steps:
1. **generate_query** - Creates search queries from user input
2. **web_research** - Performs web searches and extracts information
3. **reflection** - Evaluates if sufficient information was gathered
4. **finalize_answer** - Produces the final response with citations

Key files:
- `backend/src/agent/graph.py` - Main graph definition and workflow
- `backend/src/agent/state.py` - State type definitions
- `backend/src/agent/tools_and_schemas.py` - Web search tools and response schemas
- `backend/src/agent/prompts.py` - LLM prompt templates
- `backend/src/agent/configuration.py` - Configurable parameters (research effort levels)

### Frontend Architecture
- **Streaming UI**: Uses LangGraph SDK for real-time agent updates
- **Activity Timeline**: Visual representation of agent processing in `ActivityTimeline.tsx`
- **Component Library**: Uses Shadcn UI components in `src/components/ui/`
- **Styling**: Tailwind CSS v4 with configuration in `global.css`

### API Integration
- Frontend connects to backend via LangGraph SDK streaming
- WebSocket-based real-time updates
- State persistence using PostgreSQL
- Redis for pub-sub message broker

## Testing Approach

### Backend Testing
- Test files should be placed alongside source files or in a `tests/` directory
- Run with `uv run pytest` from the backend directory
- Tests should cover graph transitions, tool functionality, and API endpoints

### Frontend Testing
- No test framework currently set up
- When adding tests, consider using Vitest (already configured with Vite)

## Environment Configuration

Required environment variables (see `backend/.env.example`):
- `GEMINI_API_KEY` - Google AI API key for LLM access
- `LANGSMITH_API_KEY` - (Production only) For monitoring
- `POSTGRES_URI` - (Production only) Database connection
- `REDIS_URI` - (Production only) Message broker

## Docker Deployment

Production deployment uses docker-compose with:
- PostgreSQL for state persistence
- Redis for streaming/pub-sub
- Multi-stage Docker build for optimized images

Build and run:
```bash
docker-compose up --build
```

## Key Development Patterns

1. **State Management**: All agent state flows through the `AgentState` type
2. **Tool Integration**: Web search tools are defined in `tools_and_schemas.py`
3. **Prompt Engineering**: All prompts are centralized in `prompts.py`
4. **Configuration**: Research effort levels (low/medium/high) control iteration depth
5. **Error Handling**: The agent includes reflection steps to validate information quality
6. **Citation Tracking**: Sources are maintained throughout the research process