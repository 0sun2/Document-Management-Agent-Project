"""Database client initialization."""
from typing import Optional
from supabase import create_client, Client
from core.config import SUPABASE_URL, SUPABASE_KEY
from core.logging import setup_logger

logger = setup_logger(__name__)

# Supabase client
sb: Optional[Client] = None

def initialize_supabase() -> Optional[Client]:
    """Initialize Supabase client."""
    global sb

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase initialized successfully")
            return sb
        except Exception as e:
            logger.warning(f"Supabase initialization failed: {e}")
            return None
    else:
        logger.warning(
            "Supabase not configured: Add SUPABASE_URL / SUPABASE_ANON_KEY to .env"
        )
        return None


def get_supabase_client() -> Optional[Client]:
    """Get Supabase client instance."""
    return sb
