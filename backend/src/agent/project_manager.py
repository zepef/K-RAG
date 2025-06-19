"""Project management utilities for K-RAG Agent."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import json


class ProjectManager:
    """Manages project-specific prompt saving and organization."""
    
    def __init__(self, base_dir: str = None):
        """Initialize project manager with base directory for projects."""
        if base_dir is None:
            # Use absolute path to avoid resolve() which causes blocking
            base_dir = "/mnt/e/Projects/K-RAG/projects"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
    def get_project_dir(self, project_id: str, project_name: str) -> Path:
        """Get or create project directory."""
        # Create a clean directory name from project name
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        project_dir = self.base_dir / f"{clean_name}_{project_id[:8]}"
        project_dir.mkdir(exist_ok=True)
        return project_dir
        
    def save_prompt_session(
        self, 
        project_id: str,
        project_name: str,
        query: str,
        search_queries: List[Dict],
        research_results: List[Dict],
        final_answer: str,
        sources: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a complete prompt session to a markdown file."""
        project_dir = self.get_project_dir(project_id, project_name)
        
        # Generate timestamp-based filename with more readable format
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H-%M-%S")
        query_preview = "".join(c if c.isalnum() or c in "-_ " else "" for c in query[:50]).strip().replace(" ", "_")
        filename = f"{date_str}_{time_str}_{query_preview}.md"
        filepath = project_dir / filename
        
        # Create markdown content
        content = self._format_session_markdown(
            query=query,
            search_queries=search_queries,
            research_results=research_results,
            final_answer=final_answer,
            sources=sources,
            metadata=metadata
        )
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Update project index
        self._update_project_index(project_id, project_name, filepath, query)
        
        return str(filepath)
        
    def _format_session_markdown(
        self,
        query: str,
        search_queries: List[Dict],
        research_results: List[Dict],
        final_answer: str,
        sources: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Format session data as markdown."""
        lines = []
        
        # Header with timestamp
        session_timestamp = datetime.now()
        lines.append(f"# K-RAG Agent Session")
        lines.append(f"**Date:** {session_timestamp.strftime('%Y-%m-%d')}")
        lines.append(f"**Time:** {session_timestamp.strftime('%H:%M:%S')}")
        lines.append("")
        
        # Metadata
        if metadata:
            lines.append("## Metadata")
            lines.append("```json")
            lines.append(json.dumps(metadata, indent=2))
            lines.append("```")
            lines.append("")
        
        # Original Query with timestamp
        lines.append("## Original Query")
        lines.append(f"**Timestamp:** {session_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> {query}")
        lines.append("")
        
        # Search Queries
        if search_queries:
            lines.append("## Generated Search Queries")
            for i, sq in enumerate(search_queries, 1):
                lines.append(f"### Query {i}")
                lines.append(f"**Query:** {sq.get('query', 'N/A')}")
                lines.append(f"**Rationale:** {sq.get('rationale', 'N/A')}")
                lines.append("")
        
        # Research Results Summary
        if research_results:
            lines.append("## Research Results Summary")
            for i, result in enumerate(research_results, 1):
                lines.append(f"### Result {i}")
                if isinstance(result, dict):
                    lines.append(f"**Query:** {result.get('query', 'N/A')}")
                    lines.append(f"**Summary:** {result.get('summary', 'N/A')}")
                else:
                    lines.append(str(result))
                lines.append("")
        
        # Final Answer
        lines.append("## Final Answer")
        lines.append(final_answer)
        lines.append("")
        
        # Sources
        if sources:
            lines.append("## Sources")
            for source in sources:
                lines.append(f"- {source}")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("*Generated by K-RAG Agent*")
        
        return "\n".join(lines)
        
    def _update_project_index(
        self, 
        project_id: str, 
        project_name: str,
        filepath: Path,
        query: str
    ) -> None:
        """Update project index file with new session."""
        project_dir = self.get_project_dir(project_id, project_name)
        index_file = project_dir / "index.md"
        
        # Read existing content or create new
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = f"# {project_name} - Project Index\n\nProject ID: {project_id}\n\n## Sessions\n\n"
        
        # Add new session entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_entry = f"- [{timestamp}]({filepath.name}) - {query[:100]}{'...' if len(query) > 100 else ''}\n"
        
        # Insert after "## Sessions" line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == "## Sessions":
                lines.insert(i + 2, session_entry)
                break
        
        # Write updated content
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
    def list_projects(self) -> List[Dict[str, str]]:
        """List all projects with their metadata."""
        projects = []
        for project_dir in self.base_dir.iterdir():
            if project_dir.is_dir():
                index_file = project_dir / "index.md"
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract project name from first line
                        first_line = content.split('\n')[0]
                        if first_line.startswith('# '):
                            project_name = first_line[2:].split(' - ')[0]
                            projects.append({
                                'name': project_name,
                                'path': str(project_dir),
                                'sessions': len([f for f in project_dir.glob('session_*.md')])
                            })
        return projects