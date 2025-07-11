import os
import pytest

from open_deep_research.state_manager import SingleStoreStateManager


def test_singlestore_connection():
    """Test establishing a connection to SingleStoreDB."""
    dsn = os.environ.get("SINGLESTORE_URI")
    if not dsn:
        pytest.skip("SINGLESTORE_URI not set")
    try:
        manager = SingleStoreStateManager(dsn)
    except Exception as err:
        pytest.skip(f"Could not connect to SingleStoreDB: {err}")
    assert manager.conn is not None
    with manager.conn.cursor() as cur:
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1

