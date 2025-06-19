#!/usr/bin/env python3
"""Initialize Neo4j database with required indices and constraints."""

import asyncio
import os
import sys
import logging
from neo4j_manager import Neo4jManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_neo4j():
    """Initialize Neo4j database."""
    logger.info("Initializing Neo4j database...")
    
    nm = Neo4jManager()
    
    try:
        # Test connection
        async with nm.driver.session() as session:
            result = await session.run("RETURN 1 as connected")
            record = await result.single()
            if record and record["connected"] == 1:
                logger.info("Successfully connected to Neo4j")
            else:
                raise Exception("Failed to connect to Neo4j")
        
        # Create indices
        logger.info("Creating indices and constraints...")
        await nm.create_indices()
        
        logger.info("Neo4j initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")
        sys.exit(1)
    finally:
        await nm.close()


if __name__ == "__main__":
    asyncio.run(initialize_neo4j())