.PHONY: help dev-frontend dev-backend dev

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎯 K-RAG Agent - Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "  make dev             - Starts both frontend and backend development servers"
	@echo "  make dev-frontend    - Starts only the frontend development server (React/Vite)"
	@echo "  make dev-backend     - Starts only the backend development server (LangGraph)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Note: Frontend may use port 5174 if 5173 is already in use."

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend development server..."
	@echo ""
	@echo "🚀 LangGraph Studio will be available at: http://localhost:2024"
	@echo "📚 API Documentation available at: http://localhost:2024/docs"
	@echo ""
	@cd backend && ./env/bin/langgraph dev

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🎯 K-RAG Agent Development Environment"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Frontend (React):"
	@echo "  📱 Application: http://localhost:5173"
	@echo ""
	@echo "Backend (LangGraph):"
	@echo "  🚀 LangGraph Studio: http://localhost:2024"
	@echo "  📚 API Documentation: http://localhost:2024/docs"
	@echo "  🔌 API Endpoint: http://localhost:2024"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@make dev-frontend & make dev-backend 