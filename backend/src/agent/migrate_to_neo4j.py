#!/usr/bin/env python3
"""
Migration script to import existing markdown projects into Neo4j.

Usage:
    python migrate_to_neo4j.py [--projects-dir /path/to/projects]
"""

import asyncio
import argparse
from pathlib import Path
import logging
import sys
from datetime import datetime

from neo4j_manager import Neo4jManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


async def run_migration(projects_dir: Path):
    """Run the migration process."""
    logger.info(f"Starting migration from {projects_dir}")
    
    # Check if projects directory exists
    if not projects_dir.exists():
        logger.error(f"Projects directory not found: {projects_dir}")
        return
    
    # Initialize Neo4j manager
    nm = Neo4jManager()
    
    try:
        # Create indices first
        logger.info("Creating Neo4j indices...")
        await nm.create_indices()
        
        # Count projects and sessions
        project_count = sum(1 for d in projects_dir.iterdir() if d.is_dir() and not d.name.startswith('.'))
        logger.info(f"Found {project_count} projects to migrate")
        
        # Run migration
        logger.info("Starting migration process...")
        results = await nm.migrate_markdown_projects(projects_dir)
        
        logger.info(f"Migration completed!")
        logger.info(f"  - Sessions imported: {results['imported']}")
        logger.info(f"  - Sessions failed: {results['failed']}")
        
        # Get graph statistics
        stats = await nm.get_graph_statistics()
        logger.info("Graph statistics after migration:")
        logger.info(f"  - Total queries: {stats.get('total_queries', 0)}")
        logger.info(f"  - Search queries: {stats.get('search_queries', {})}")
        logger.info(f"  - Total content: {stats.get('total_content', 0)}")
        logger.info(f"  - Total webpages: {stats.get('total_webpages', 0)}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
    finally:
        await nm.close()


async def verify_migration(projects_dir: Path):
    """Verify the migration by checking a few random sessions."""
    logger.info("Verifying migration...")
    
    nm = Neo4jManager()
    
    try:
        # Get a sample of projects
        sample_projects = []
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                parts = project_dir.name.rsplit('_', 1)
                if len(parts) == 2:
                    sample_projects.append({
                        'name': parts[0].replace('_', ' '),
                        'id': parts[1],
                        'dir': project_dir
                    })
                    if len(sample_projects) >= 3:  # Check up to 3 projects
                        break
        
        for project in sample_projects:
            logger.info(f"Checking project: {project['name']}")
            
            # Count markdown files
            md_files = list(project['dir'].glob('*.md'))
            session_count = len([f for f in md_files if f.name != 'index.md'])
            
            # Query Neo4j for this project's sessions
            async with nm.driver.session() as session:
                result = await session.run("""
                    MATCH (q:Query {project_id: $project_id})
                    RETURN COUNT(q) as count
                """, project_id=project['id'])
                
                record = await result.single()
                neo4j_count = record['count'] if record else 0
                
                logger.info(f"  - Markdown sessions: {session_count}")
                logger.info(f"  - Neo4j queries: {neo4j_count}")
                
                if session_count != neo4j_count:
                    logger.warning(f"  - Mismatch detected! Expected {session_count}, got {neo4j_count}")
                else:
                    logger.info("  - ✓ Counts match")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
    finally:
        await nm.close()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Migrate K-RAG markdown projects to Neo4j')
    parser.add_argument(
        '--projects-dir',
        type=Path,
        default=Path('/mnt/e/Projects/K-RAG/projects'),
        help='Path to the projects directory'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify the migration without running it'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        await verify_migration(args.projects_dir)
    else:
        await run_migration(args.projects_dir)
        logger.info("\nRunning verification...")
        await verify_migration(args.projects_dir)


if __name__ == "__main__":
    asyncio.run(main())