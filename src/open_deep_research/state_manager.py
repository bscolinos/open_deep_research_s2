import json
from typing import Any, Dict, Optional, Tuple

try:
    import singlestoredb as s2
except Exception:  # pragma: no cover - optional dependency may be missing
    s2 = None

class SingleStoreStateManager:
    """Persist report and section state to SingleStoreDB tables."""

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn
        self.conn = None
        self.report_states: Dict[str, Dict[str, Any]] = {}
        self.section_states: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if dsn and s2 is not None:
            self.conn = s2.connect(dsn)
            self._init_tables()

    def _init_tables(self) -> None:
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS report_state ("
                "thread_id VARCHAR(64) PRIMARY KEY,"
                "state JSON"
                ")"
            )
            cur.execute(
                "CREATE TABLE IF NOT EXISTS section_state ("
                "thread_id VARCHAR(64),"
                "section_name VARCHAR(255),"
                "state JSON,"
                "PRIMARY KEY(thread_id, section_name)"
                ")"
            )
        self.conn.commit()

    # ------------------------ Report state ---------------------
    def save_report_state(self, thread_id: str, state: Dict[str, Any]) -> None:
        if self.conn is not None:
            with self.conn.cursor() as cur:
                cur.execute(
                    "REPLACE INTO report_state (thread_id, state) VALUES (%s, %s)",
                    (thread_id, json.dumps(state)),
                )
            self.conn.commit()
        else:
            self.report_states[thread_id] = state.copy()

    def load_report_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        if self.conn is not None:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT state FROM report_state WHERE thread_id=%s",
                    (thread_id,),
                )
                row = cur.fetchone()
            return json.loads(row[0]) if row else None
        return self.report_states.get(thread_id)

    # ------------------------ Section state --------------------
    def save_section_state(
        self, thread_id: str, section_name: str, state: Dict[str, Any]
    ) -> None:
        if self.conn is not None:
            with self.conn.cursor() as cur:
                cur.execute(
                    "REPLACE INTO section_state (thread_id, section_name, state)"
                    " VALUES (%s, %s, %s)",
                    (thread_id, section_name, json.dumps(state)),
                )
            self.conn.commit()
        else:
            self.section_states[(thread_id, section_name)] = state.copy()

    def load_section_state(
        self, thread_id: str, section_name: str
    ) -> Optional[Dict[str, Any]]:
        if self.conn is not None:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT state FROM section_state WHERE thread_id=%s AND section_name=%s",
                    (thread_id, section_name),
                )
                row = cur.fetchone()
            return json.loads(row[0]) if row else None
        return self.section_states.get((thread_id, section_name))
