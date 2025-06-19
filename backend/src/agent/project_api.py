"""API endpoints for project management."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
from pathlib import Path
import json
from datetime import datetime

from agent.project_manager import ProjectManager
import asyncio

router = APIRouter(prefix="/projects", tags=["projects"])

class ProjectInfo(BaseModel):
    """Project information model."""
    id: str
    name: str
    created_at: Optional[str] = None
    session_count: int = 0
    last_updated: Optional[str] = None

class ProjectSession(BaseModel):
    """Project session summary."""
    filename: str
    timestamp: str
    query: str
    path: str

@router.get("/list", response_model=List[ProjectInfo])
async def list_projects():
    """List all available projects."""
    # Run blocking operations in thread pool
    def _list_projects():
        project_manager = ProjectManager()
        projects_dir = project_manager.base_dir
        
        projects = []
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith('.'):
                    # Parse project name and ID from directory name
                    parts = project_dir.name.rsplit('_', 1)
                    if len(parts) == 2:
                        project_name = parts[0].replace('_', ' ')
                        project_id = parts[1]
                        
                        # Count sessions
                        session_files = list(project_dir.glob('*.md'))
                        session_count = len([f for f in session_files if f.name != 'index.md'])
                        
                        # Get timestamps
                        created_at = None
                        last_updated = None
                        
                        if session_files:
                            # Get creation time from oldest file
                            oldest = min(session_files, key=lambda f: f.stat().st_ctime)
                            created_at = datetime.fromtimestamp(oldest.stat().st_ctime).isoformat()
                            
                            # Get last update from newest file
                            newest = max(session_files, key=lambda f: f.stat().st_mtime)
                            last_updated = datetime.fromtimestamp(newest.stat().st_mtime).isoformat()
                        
                        projects.append(ProjectInfo(
                            id=project_id,
                            name=project_name,
                            created_at=created_at,
                            session_count=session_count,
                            last_updated=last_updated
                        ))
        
        return sorted(projects, key=lambda p: p.last_updated or '', reverse=True)
    
    return await asyncio.to_thread(_list_projects)

@router.get("/{project_id}/sessions", response_model=List[ProjectSession])
async def get_project_sessions(project_id: str):
    """Get all sessions for a specific project."""
    def _get_sessions():
        project_manager = ProjectManager()
        
        # Find the project directory
        project_dir = None
        for d in project_manager.base_dir.iterdir():
            if d.is_dir() and d.name.endswith(f"_{project_id}"):
                project_dir = d
                break
        
        if not project_dir:
            raise HTTPException(status_code=404, detail="Project not found")
        
        sessions = []
        for session_file in project_dir.glob('*.md'):
            if session_file.name == 'index.md':
                continue
                
            # Parse filename for timestamp and query
            parts = session_file.stem.split('_', 2)
            if len(parts) >= 3:
                date_str = parts[0]
                time_str = parts[1]
                query_preview = parts[2].replace('_', ' ')
                
                # Read first few lines to get actual query
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        actual_query = query_preview
                        for i, line in enumerate(lines):
                            if line.strip().startswith("> ") and i > 0 and "Original Query" in lines[i-2]:
                                actual_query = line.strip()[2:]
                                break
                except:
                    actual_query = query_preview
                
                sessions.append(ProjectSession(
                    filename=session_file.name,
                    timestamp=f"{date_str} {time_str.replace('-', ':')}",
                    query=actual_query,
                    path=str(session_file)
                ))
        
        return sorted(sessions, key=lambda s: s.timestamp, reverse=True)
    
    return await asyncio.to_thread(_get_sessions)

@router.get("/{project_id}/details")
async def get_project_details(project_id: str):
    """Get detailed information about a project including its full name."""
    def _get_details():
        project_manager = ProjectManager()
        
        # Find the project directory
        project_dir = None
        project_name = None
        for d in project_manager.base_dir.iterdir():
            if d.is_dir() and d.name.endswith(f"_{project_id}"):
                project_dir = d
                # Extract project name from directory
                parts = d.name.rsplit('_', 1)
                if len(parts) == 2:
                    project_name = parts[0].replace('_', ' ')
                break
        
        if not project_dir:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Read index file if exists
        index_file = project_dir / "index.md"
        description = ""
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract description or first paragraph
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('#') and not line.startswith('Project ID:'):
                            description = line.strip()
                            break
            except:
                pass
        
        return {
            "id": project_id,
            "name": project_name,
            "path": str(project_dir),
            "description": description
        }
    
    return await asyncio.to_thread(_get_details)