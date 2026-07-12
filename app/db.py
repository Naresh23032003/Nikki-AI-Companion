"""SQLite persistence for sessions and messages.

Schema:
  sessions(id TEXT PRIMARY KEY, created_at TEXT)
  messages(id INTEGER PK, session_id TEXT, role TEXT, content TEXT, timestamp TEXT)

We keep a single connection guarded by a lock. SQLite writes are fast and the
app is single-process, so this is plenty for a local companion.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_occurrence(due_at_iso: str, recurrence_rule: str) -> str:
    """Roll a recurring reminder's due_at forward by one period.

    Supported rules: "daily", "weekly", or "every <weekday>" (e.g. "every monday").
    Unknown rules default to +1 day rather than silently never re-firing.
    """
    try:
        due = datetime.fromisoformat(due_at_iso)
    except ValueError:
        due = datetime.now(timezone.utc)
    rule = (recurrence_rule or "").strip().lower()
    if rule == "weekly" or rule.startswith("every "):
        return (due + timedelta(days=7)).isoformat()
    return (due + timedelta(days=1)).isoformat()  # "daily" and fallback


class Database:
    def __init__(self, path: Path):
        self.path = path
        # check_same_thread=False because FastAPI may touch the connection from
        # different threads; we serialize access ourselves with a lock.
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL + NORMAL sync: readers don't block writers and commits skip the
        # full fsync-per-write default, which matters here because a single
        # /chat turn does 4+ writes (user msg, exchange_count, settings,
        # assistant msg) all in the request path before generation even starts.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    timestamp  TEXT NOT NULL,
                    audio_url  TEXT,
                    sticker_url TEXT,
                    source     TEXT NOT NULL DEFAULT 'webapp_chat',
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);

                -- messages_between() (nightly mood-journal extraction) scans
                -- by timestamp across ALL sessions — table scan without this.
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                    ON messages(timestamp);

                CREATE TABLE IF NOT EXISTS memories (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact            TEXT NOT NULL,
                    category        TEXT NOT NULL,
                    kind            TEXT NOT NULL DEFAULT 'permanent',
                    event_datetime  TEXT,
                    recurrence_rule TEXT,
                    valid_from      TEXT,
                    valid_until     TEXT,
                    source          TEXT,
                    created_at      TEXT NOT NULL,
                    last_accessed   TEXT,
                    access_count    INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_memories_created
                    ON memories(created_at DESC);

                CREATE TABLE IF NOT EXISTS entities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    type            TEXT NOT NULL DEFAULT 'thing',
                    notes           TEXT,
                    created_at      TEXT NOT NULL,
                    last_seen       TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_entities_normalized
                    ON entities(normalized_name);

                CREATE TABLE IF NOT EXISTS relations (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id     INTEGER NOT NULL,
                    relation      TEXT NOT NULL,
                    target_id     INTEGER NOT NULL,
                    confidence    REAL NOT NULL DEFAULT 0.5,
                    created_at    TEXT NOT NULL,
                    last_confirmed TEXT NOT NULL,
                    ended_at      TEXT,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                );

                CREATE INDEX IF NOT EXISTS idx_relations_active
                    ON relations(source_id, target_id, ended_at);

                CREATE TABLE IF NOT EXISTS app_settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS reminders (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    text            TEXT NOT NULL,
                    due_at          TEXT NOT NULL,
                    delivered       INTEGER NOT NULL DEFAULT 0,
                    recurrence_rule TEXT,
                    cancelled       INTEGER NOT NULL DEFAULT 0,
                    session_id      TEXT,
                    created_at      TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_call_log (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT NOT NULL,
                    args       TEXT,
                    ok         INTEGER NOT NULL,
                    result     TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event_followups (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id          INTEGER,
                    event_fact         TEXT NOT NULL,
                    event_datetime     TEXT NOT NULL,
                    encouragement_at   TEXT,
                    followup_at        TEXT NOT NULL,
                    encouragement_sent INTEGER NOT NULL DEFAULT 0,
                    followup_sent      INTEGER NOT NULL DEFAULT 0,
                    awaiting_answer    INTEGER NOT NULL DEFAULT 0,
                    resolved           INTEGER NOT NULL DEFAULT 0,
                    session_id         TEXT,
                    created_at         TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mood_journal (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    date            TEXT NOT NULL,
                    time            TEXT,
                    mood_label      TEXT NOT NULL,
                    intensity       INTEGER NOT NULL,
                    why             TEXT NOT NULL,
                    source_channel  TEXT,
                    created_at      TEXT NOT NULL,
                    edited          INTEGER NOT NULL DEFAULT 0,
                    synced_to_sheets INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_mood_journal_date ON mood_journal(date);

                CREATE TABLE IF NOT EXISTS device_presence (
                    device_id  TEXT PRIMARY KEY,
                    kind       TEXT NOT NULL,
                    last_seen  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deferred_tasks (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind       TEXT NOT NULL,
                    question   TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    not_before TEXT,
                    attempts   INTEGER NOT NULL DEFAULT 0,
                    done       INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS day_state (
                    day    TEXT PRIMARY KEY,
                    state  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS relationship_state (
                    id                   INTEGER PRIMARY KEY CHECK (id = 1),
                    stage                TEXT NOT NULL DEFAULT 'stranger',
                    affection            REAL NOT NULL DEFAULT 5.0,
                    started_at           TEXT NOT NULL,
                    days_known           INTEGER NOT NULL DEFAULT 0,
                    meaningful_exchanges INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            # Migrations for pre-existing databases.
            self._conn.execute("DROP TABLE IF EXISTS todos")
            self._conn.execute("DROP TABLE IF EXISTS expenses")
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(messages)")}
            if "audio_url" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN audio_url TEXT")
            if "sticker_url" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN sticker_url TEXT")
            if "source" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN source TEXT NOT NULL DEFAULT 'webapp_chat'")
            self._conn.execute("UPDATE messages SET source = 'webapp_chat' WHERE source IS NULL OR source = ''")

            mem_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(memories)")}
            for column, ddl in [
                ("kind", "TEXT NOT NULL DEFAULT 'permanent'"),
                ("event_datetime", "TEXT"),
                ("recurrence_rule", "TEXT"),
                ("valid_from", "TEXT"),
                ("valid_until", "TEXT"),
                ("source", "TEXT"),
            ]:
                if column not in mem_cols:
                    self._conn.execute(f"ALTER TABLE memories ADD COLUMN {column} {ddl}")
            self._conn.execute("UPDATE memories SET kind = 'permanent' WHERE kind IS NULL OR kind = ''")
            self._conn.execute("UPDATE memories SET source = 'webapp_chat' WHERE source IS NULL OR source = ''")

            rem_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(reminders)")}
            for column, ddl in [
                ("recurrence_rule", "TEXT"),
                ("cancelled", "INTEGER NOT NULL DEFAULT 0"),
                ("session_id", "TEXT"),
            ]:
                if column not in rem_cols:
                    self._conn.execute(f"ALTER TABLE reminders ADD COLUMN {column} {ddl}")
            self._conn.commit()

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    @staticmethod
    def _normalize_source(source: str | None, session_id: str | None = None) -> str:
        allowed = {"webapp_chat", "webapp_call", "whatsapp", "tablet", "iot"}
        raw = (source or "").strip().lower()
        if raw in allowed:
            return raw
        if session_id:
            sid = str(session_id).lower()
            if sid.startswith("tablet") or sid.startswith("tablet:"):
                return "tablet"
            if sid.startswith("iot") or sid.startswith("iot:"):
                return "iot"
            if sid.startswith("wa") or "whatsapp" in sid:
                return "whatsapp"
            if sid.startswith("call") or sid.startswith("webapp_call"):
                return "webapp_call"
        return "webapp_chat"

    # Nouns that suggest a dated event + words that anchor it to a moment.
    _EVENT_NOUN = re.compile(
        r"\b(birthday|anniversary|exam|test|appointment|meeting|interview|"
        r"trip|flight|call|deadline|party|wedding|concert|date)\b", re.I)
    # One-off dated events (excludes birthday/anniversary, which recur yearly
    # and must NOT age out). Used for the event_datetime-based promotion below.
    _ONEOFF_EVENT_NOUN = re.compile(
        r"\b(exam|test|appointment|meeting|interview|trip|flight|call|"
        r"deadline|party|wedding|concert|date)\b", re.I)
    # Yearly-recurring nouns: their presence blocks the one-off datetime
    # promotion even when a one-off word also appears ("wedding anniversary"
    # matches "wedding" but is really a yearly anniversary).
    _RECURRING_EVENT_NOUN = re.compile(r"\b(birthday|anniversary)\b", re.I)
    _TIME_MARKER = re.compile(
        r"\b(today|tomorrow|tonight|this (morning|afternoon|evening|week(end)?)|"
        r"next (week|month|mon|tue|wed|thu|fri|sat|sun)\w*|"
        r"on (mon|tue|wed|thu|fri|sat|sun)\w*|at \d{1,2})\b", re.I)

    @staticmethod
    def _normalize_kind(kind: str | None, fact: str | None = None,
                        event_datetime: str | None = None,
                        recurrence_rule: str | None = None) -> str:
        raw = (kind or "permanent").strip().lower()
        has_event_noun = bool(fact and Database._EVENT_NOUN.search(fact))
        has_rule = bool((recurrence_rule or "").strip())
        # Time-anchored either by a word in the fact TEXT, or by a resolved
        # event_datetime. The datetime path is scoped to ONE-OFF event nouns:
        # the extractor routinely resolves event_datetime correctly while
        # dropping the time word from the fact string ("User has a dentist
        # appointment" + event_datetime set), which used to leave these as
        # "permanent" (date then NULLED, injected forever) or "recurring"
        # (never ages out). birthday/anniversary are excluded so a dated
        # yearly event isn't wrongly aged out.
        anchored_by_text = bool(fact and Database._TIME_MARKER.search(fact))
        anchored_by_dt = (
            bool(event_datetime)
            and bool(fact and Database._ONEOFF_EVENT_NOUN.search(fact))
            and not bool(fact and Database._RECURRING_EVENT_NOUN.search(fact)))
        anchored = anchored_by_text or anchored_by_dt
        if has_event_noun and anchored:
            if raw in {"permanent", "event"} or raw not in {"recurring", "transient"}:
                return "event"
            # "recurring" with a concrete date but NO recurrence rule is really
            # a one-off event mislabeled; a genuine recurring fact has a rule.
            if raw == "recurring" and not has_rule:
                return "event"
        if raw in {"permanent", "event", "recurring", "transient"}:
            return raw
        if has_event_noun:
            return "event"
        return "permanent"

    @staticmethod
    def _infer_temporal_defaults(fact: str, category: str, kind: str | None, event_datetime: str | None, recurrence_rule: str | None, valid_from: str | None, valid_until: str | None) -> tuple[str, str | None, str | None, str | None, str | None]:
        kind = Database._normalize_kind(kind, fact, event_datetime, recurrence_rule)
        now = datetime.now(timezone.utc)
        if kind == "event" and not event_datetime:
            text = fact.lower()
            # "at 12" means 12:00 in the USER's local timezone — resolve locally,
            # store as UTC (previously this made "at 12" mean 12:00 UTC).
            now_local = datetime.now().astimezone()
            m = re.search(r"\b(at|@)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
            if m:
                hour = int(m.group(2))
                minute = int(m.group(3) or 0)
                meridiem = m.group(4)
                if meridiem == "pm" and hour < 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                local_dt = now_local.replace(hour=min(hour, 23), minute=minute,
                                             second=0, microsecond=0)
            elif "tonight" in text:
                local_dt = now_local.replace(hour=20, minute=0, second=0, microsecond=0)
            else:
                # No stated time: default to noon local rather than "right now".
                local_dt = now_local.replace(hour=12, minute=0, second=0, microsecond=0)
            # Day resolution: "tomorrow" (and "today"/default = today).
            if "tomorrow" in text:
                local_dt += timedelta(days=1)
            event_datetime = local_dt.astimezone(timezone.utc).isoformat()
        if kind == "transient" and not valid_until:
            valid_until = (now + timedelta(hours=12)).isoformat()
        if kind == "event" and not valid_from:
            valid_from = now.isoformat()
        if kind == "event" and not valid_until and event_datetime:
            valid_until = event_datetime
        if kind == "recurring" and not valid_from:
            valid_from = now.isoformat()
        if kind == "permanent":
            event_datetime = None
            recurrence_rule = None
        return kind, event_datetime, recurrence_rule, valid_from, valid_until

    def upsert_entity(self, name: str, entity_type: str = "thing", notes: str | None = None) -> int:
        normalized = self._normalize_name(name)
        if not normalized:
            normalized = "thing"
        entity_type = entity_type if entity_type in {"person", "place", "thing", "event", "pet", "org"} else "thing"
        now = _utcnow()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, notes FROM entities WHERE normalized_name = ? ORDER BY id DESC LIMIT 1",
                (normalized,),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    "UPDATE entities SET name = ?, type = ?, notes = ?, last_seen = ? WHERE id = ?",
                    (name.strip(), entity_type, notes or row["notes"] or "", now, row["id"]),
                )
                self._conn.commit()
                return int(row["id"])

            rows = self._conn.execute("SELECT id, name, normalized_name FROM entities").fetchall()
            best_row = None
            best_score = 0.0
            for candidate in rows:
                score = SequenceMatcher(None, normalized, candidate["normalized_name"]).ratio()
                if score >= 0.82 and score > best_score:
                    best_row = candidate
                    best_score = score
            if best_row is not None:
                self._conn.execute(
                    "UPDATE entities SET name = ?, type = ?, notes = ?, last_seen = ? WHERE id = ?",
                    (name.strip(), entity_type, notes or "", now, best_row["id"]),
                )
                self._conn.commit()
                return int(best_row["id"])

            cur = self._conn.execute(
                "INSERT INTO entities (name, normalized_name, type, notes, created_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                (name.strip(), normalized, entity_type, notes or "", now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_entity(self, entity_id: int) -> Dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM entities WHERE id = ?", (entity_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_entities(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_entity(self, entity_id: int, name: str | None = None, entity_type: str | None = None, notes: str | None = None) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                return False
            updates = []
            values = []
            if name is not None:
                updates.append("name = ?")
                values.append(name.strip())
                updates.append("normalized_name = ?")
                values.append(self._normalize_name(name))
            if entity_type is not None:
                updates.append("type = ?")
                values.append(entity_type if entity_type in {"person", "place", "thing", "event", "pet", "org"} else "thing")
            if notes is not None:
                updates.append("notes = ?")
                values.append(notes)
            updates.append("last_seen = ?")
            values.append(_utcnow())
            values.append(entity_id)
            self._conn.execute(f"UPDATE entities SET {', '.join(updates)} WHERE id = ?", values)
            self._conn.commit()
            return True

    def delete_entity(self, entity_id: int) -> bool:
        with self._lock:
            self._conn.execute("DELETE FROM relations WHERE source_id = ? OR target_id = ?", (entity_id, entity_id))
            cur = self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def upsert_relation(self, source_id: int, relation: str, target_id: int, confidence: float = 0.5) -> int:
        if source_id == target_id:
            return -1
        relation = relation.strip()
        now = _utcnow()
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM relations WHERE source_id = ? AND relation = ? AND target_id = ? AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
                (source_id, relation, target_id),
            ).fetchone()
            if existing is not None:
                self._conn.execute(
                    "UPDATE relations SET confidence = ?, last_confirmed = ? WHERE id = ?",
                    (min(1.0, confidence), now, existing["id"]),
                )
                self._conn.commit()
                return int(existing["id"])

            prior = self._conn.execute(
                "SELECT id FROM relations WHERE ((source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)) AND ended_at IS NULL ORDER BY id DESC LIMIT 1",
                (source_id, target_id, target_id, source_id),
            ).fetchone()
            if prior is not None:
                self._conn.execute(
                    "UPDATE relations SET ended_at = ? WHERE id = ?",
                    (now, prior["id"]),
                )

            cur = self._conn.execute(
                "INSERT INTO relations (source_id, relation, target_id, confidence, created_at, last_confirmed, ended_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (source_id, relation, target_id, min(1.0, confidence), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_relation(self, relation_id: int) -> Dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM relations WHERE id = ?", (relation_id,)).fetchone()
        return dict(row) if row else None

    def list_relations(self, active_only: bool = True) -> List[Dict]:
        query = "SELECT * FROM relations"
        if active_only:
            query += " WHERE ended_at IS NULL"
        query += " ORDER BY created_at DESC, id DESC"
        with self._lock:
            rows = self._conn.execute(query).fetchall()
        return [dict(r) for r in rows]

    def get_relations_for_entity(self, entity_id: int, active_only: bool = True) -> List[Dict]:
        query = "SELECT * FROM relations WHERE source_id = ? OR target_id = ?"
        if active_only:
            query += " AND ended_at IS NULL"
        query += " ORDER BY created_at DESC"
        with self._lock:
            rows = self._conn.execute(query, (entity_id, entity_id)).fetchall()
        return [dict(r) for r in rows]

    def delete_relation(self, relation_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def ensure_session(self, session_id: str) -> None:
        """Create the session row if it doesn't already exist."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
                (session_id, _utcnow()),
            )
            self._conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        audio_url: str | None = None,
        sticker_url: str | None = None,
        source: str | None = None,
    ) -> int:
        """Insert a message; returns its new id."""
        source = self._normalize_source(source, session_id)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages "
                "(session_id, role, content, timestamp, audio_url, sticker_url, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, _utcnow(), audio_url, sticker_url, source),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def set_message_audio(self, message_id: int, audio_url: str) -> None:
        """Attach a synthesized voice-note URL to a stored message."""
        with self._lock:
            self._conn.execute(
                "UPDATE messages SET audio_url = ? WHERE id = ?",
                (audio_url, message_id),
            )
            self._conn.commit()

    def get_recent_messages(self, session_id: str, limit: int) -> List[Dict[str, str]]:
        """Return the last `limit` messages for a session in chronological order.

        Each item is {"role": ..., "content": ...} ready for the LLM API.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? AND role IN ('user', 'assistant') "
                "AND content != '' "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        # rows are newest-first; reverse to chronological order. Non-conversational
        # rows (event markers, sticker-only messages) are excluded so the LLM
        # never sees them.
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_all_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Full history for a session (used for inspection / debugging)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, timestamp, audio_url, sticker_url, source "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "audio_url": r["audio_url"],
                "sticker_url": r["sticker_url"],
                "source": r["source"],
            }
            for r in rows
        ]

    def messages_between(self, start_iso: str, end_iso: str) -> List[Dict]:
        """All real (non-event) messages across every session/source in a UTC
        timestamp range — used by the nightly mood-journal extraction, which
        needs the whole day's conversation regardless of which surface
        (web/call/WhatsApp/tablet) it happened on."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, role, content, timestamp, source FROM messages "
                "WHERE role IN ('user', 'assistant') AND content != '' "
                "AND timestamp >= ? AND timestamp < ? ORDER BY id",
                (start_iso, end_iso),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_activity(self, session_id: str) -> Dict:
        """Timestamps + last texts used by the proactive scheduler."""
        with self._lock:
            row_u = self._conn.execute(
                "SELECT content, timestamp FROM messages WHERE session_id = ? "
                "AND role = 'user' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            row_a = self._conn.execute(
                "SELECT content, timestamp FROM messages WHERE session_id = ? "
                "AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return {
            "last_user_text": row_u["content"] if row_u else None,
            "last_user_ts": row_u["timestamp"] if row_u else None,
            "last_assistant_ts": row_a["timestamp"] if row_a else None,
        }

    # ------------------------------------------------------------------
    # Long-term memory rows (vectors live in ChromaDB; see app/memory.py)
    # ------------------------------------------------------------------
    def add_memory(
        self,
        fact: str,
        category: str,
        kind: str | None = None,
        event_datetime: str | None = None,
        recurrence_rule: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        source: str | None = None,
    ) -> int:
        """Insert a memory row and return its new id."""
        kind, event_datetime, recurrence_rule, valid_from, valid_until = self._infer_temporal_defaults(
            fact, category, kind, event_datetime, recurrence_rule, valid_from, valid_until
        )
        now = _utcnow()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (fact, category, kind, event_datetime, recurrence_rule, valid_from, valid_until, source, created_at, last_accessed, access_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (fact, category, kind, event_datetime, recurrence_rule, valid_from, valid_until, self._normalize_source(source), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_memory(
        self,
        memory_id: int,
        fact: str,
        category: str,
        kind: str | None = None,
        event_datetime: str | None = None,
        recurrence_rule: str | None = None,
        valid_from: str | None = None,
        valid_until: str | None = None,
        source: str | None = None,
    ) -> None:
        """Overwrite an existing memory's fact/category (dedup merge)."""
        kind, event_datetime, recurrence_rule, valid_from, valid_until = self._infer_temporal_defaults(
            fact, category, kind, event_datetime, recurrence_rule, valid_from, valid_until
        )
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET fact = ?, category = ?, kind = ?, event_datetime = ?, recurrence_rule = ?, valid_from = ?, valid_until = ?, source = ?, last_accessed = ? WHERE id = ?",
                (fact, category, kind, event_datetime, recurrence_rule, valid_from, valid_until, self._normalize_source(source), _utcnow(), memory_id),
            )
            self._conn.commit()

    def touch_memories(self, memory_ids: List[int]) -> None:
        """Bump last_accessed + access_count for retrieved memories."""
        if not memory_ids:
            return
        now = _utcnow()
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET last_accessed = ?, "
                "access_count = access_count + 1 WHERE id = ?",
                [(now, mid) for mid in memory_ids],
            )
            self._conn.commit()

    def complete_memory(self, memory_id: int) -> bool:
        """Mark an event/transient memory as done — it stops being injected.

        Uses an epoch valid_until sentinel, which _memory_is_current treats as
        'completed' regardless of whether the event date is past or future.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET valid_until = '1970-01-01T00:00:00+00:00' "
                "WHERE id = ?",
                (memory_id,),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_memories_by_ids(self, memory_ids: List[int]) -> List[Dict]:
        """Fetch multiple memory rows in one query (retrieval hot path)."""
        if not memory_ids:
            return []
        marks = ",".join("?" * len(memory_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE id IN ({marks})", memory_ids
            ).fetchall()
        return [dict(r) for r in rows]

    def get_memory(self, memory_id: int) -> Dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_memories(self) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_memories_by_category(self, category: str) -> List[Dict]:
        """Memories filtered at the SQL level (vs. list_memories() + a Python
        filter) — used by the pattern-note hot path, which runs on every
        message but usually returns nothing (eagerness-gated)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY created_at DESC, id DESC",
                (category,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_memories(self, limit: int) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Relationship state (single row; see app/relationship.py)
    # ------------------------------------------------------------------
    def get_relationship(self) -> Dict:
        """Return the relationship row, creating it on first access."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM relationship_state WHERE id = 1"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO relationship_state (id, stage, affection, "
                    "started_at, days_known, meaningful_exchanges) "
                    "VALUES (1, 'stranger', 5.0, ?, 0, 0)",
                    (_utcnow(),),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT * FROM relationship_state WHERE id = 1"
                ).fetchone()
        return dict(row)

    def update_relationship(self, **fields) -> None:
        """Update given columns of the single relationship row."""
        allowed = {"stage", "affection", "days_known", "meaningful_exchanges"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        sets = ", ".join(f"{c} = ?" for c in cols)
        with self._lock:
            self._conn.execute(
                f"UPDATE relationship_state SET {sets} WHERE id = 1",
                [fields[c] for c in cols],
            )
            self._conn.commit()

    def count_memories(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"])

    # ------------------------------------------------------------------
    # Tool tables: reminders / deferred tasks / day state
    # ------------------------------------------------------------------
    def add_reminder(self, text: str, due_at: str, recurrence_rule: str | None = None,
                     session_id: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reminders (text, due_at, recurrence_rule, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (text, due_at, recurrence_rule, session_id, _utcnow()))
            self._conn.commit()
            return int(cur.lastrowid)

    def due_reminders(self, now_iso: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE delivered = 0 AND cancelled = 0 AND due_at <= ?",
                (now_iso,)).fetchall()
        return [dict(r) for r in rows]

    def mark_reminder_delivered(self, rid: int) -> None:
        """Mark delivered. Recurring reminders instead roll to their next
        occurrence (delivered stays 0 so due_reminders() picks it up again)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT due_at, recurrence_rule FROM reminders WHERE id = ?", (rid,)
            ).fetchone()
            if row and row["recurrence_rule"]:
                next_due = _next_occurrence(row["due_at"], row["recurrence_rule"])
                self._conn.execute(
                    "UPDATE reminders SET due_at = ? WHERE id = ?", (next_due, rid))
            else:
                self._conn.execute(
                    "UPDATE reminders SET delivered = 1 WHERE id = ?", (rid,))
            self._conn.commit()

    def list_reminders(self, pending_only: bool = True) -> List[Dict]:
        q = "SELECT * FROM reminders WHERE cancelled = 0"
        if pending_only:
            q += " AND delivered = 0"
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY due_at").fetchall()
        return [dict(r) for r in rows]

    def cancel_reminder_like(self, text_like: str) -> Dict | None:
        """Cancel the most relevant pending reminder matching `text_like`
        (fuzzy substring match — the model can't track numeric IDs reliably)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE delivered = 0 AND cancelled = 0"
            ).fetchall()
            best, best_score = None, 0.0
            needle = text_like.lower().strip()
            for row in rows:
                score = SequenceMatcher(None, needle, row["text"].lower()).ratio()
                if needle in row["text"].lower():
                    score += 0.3
                if score > best_score:
                    best, best_score = row, score
            if best is None or best_score < 0.3:
                return None
            self._conn.execute("UPDATE reminders SET cancelled = 1 WHERE id = ?", (best["id"],))
            self._conn.commit()
            return dict(best)

    def add_deferred(self, kind: str, question: str, session_id: str,
                     not_before: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO deferred_tasks (kind, question, session_id, not_before, "
                "created_at) VALUES (?, ?, ?, ?, ?)",
                (kind, question, session_id, not_before, _utcnow()))
            self._conn.commit()
            return int(cur.lastrowid)

    def pending_deferred(self, now_iso: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM deferred_tasks WHERE done = 0 AND attempts < 5 "
                "AND (not_before IS NULL OR not_before <= ?) ORDER BY id", (now_iso,)
            ).fetchall()
        return [dict(r) for r in rows]

    def update_deferred(self, task_id: int, done: bool = False,
                        bump_attempts: bool = False, not_before: str | None = None):
        with self._lock:
            if done:
                self._conn.execute(
                    "UPDATE deferred_tasks SET done = 1 WHERE id = ?", (task_id,))
            if bump_attempts:
                self._conn.execute(
                    "UPDATE deferred_tasks SET attempts = attempts + 1, "
                    "not_before = ? WHERE id = ?", (not_before, task_id))
            self._conn.commit()

    def list_deferred(self, limit: int = 10) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM deferred_tasks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_day_state(self, day: str) -> Dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM day_state WHERE day = ?", (day,)).fetchone()
        if not row:
            return None
        try:
            import json as _json
            return _json.loads(row["state"])
        except Exception:  # noqa: BLE001
            return None

    def set_day_state(self, day: str, state: Dict) -> None:
        import json as _json
        with self._lock:
            self._conn.execute(
                "INSERT INTO day_state (day, state) VALUES (?, ?) "
                "ON CONFLICT(day) DO UPDATE SET state = excluded.state",
                (day, _json.dumps(state)))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Key/value app settings (e.g. runtime-selected active persona)
    # ------------------------------------------------------------------
    def get_setting(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Session maintenance
    # ------------------------------------------------------------------
    def clear_session(self, session_id: str) -> int:
        """Delete all messages for a session. Returns rows removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Tool-call logging (7.13 tool-calling framework)
    # ------------------------------------------------------------------
    def log_tool_call(self, name: str, args: str, ok: bool, result: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tool_call_log (name, args, ok, result, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (name, args, int(ok), result, _utcnow()))
            self._conn.commit()

    def recent_tool_calls(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tool_call_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Event-aware follow-ups (care check-ins around dated events)
    # ------------------------------------------------------------------
    def add_event_followup(self, memory_id: int | None, event_fact: str,
                           event_datetime: str, encouragement_at: str | None,
                           followup_at: str, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO event_followups (memory_id, event_fact, event_datetime, "
                "encouragement_at, followup_at, session_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (memory_id, event_fact, event_datetime, encouragement_at,
                 followup_at, session_id, _utcnow()))
            self._conn.commit()
            return int(cur.lastrowid)

    def due_encouragements(self, now_iso: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM event_followups WHERE resolved = 0 AND "
                "encouragement_sent = 0 AND encouragement_at IS NOT NULL AND "
                "encouragement_at <= ?", (now_iso,)).fetchall()
        return [dict(r) for r in rows]

    def due_followups(self, now_iso: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM event_followups WHERE resolved = 0 AND "
                "followup_sent = 0 AND followup_at <= ?", (now_iso,)).fetchall()
        return [dict(r) for r in rows]

    def mark_encouragement_sent(self, fid: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE event_followups SET encouragement_sent = 1 WHERE id = ?", (fid,))
            self._conn.commit()

    def mark_followup_sent(self, fid: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE event_followups SET followup_sent = 1, awaiting_answer = 1 "
                "WHERE id = ?", (fid,))
            self._conn.commit()

    def get_awaiting_followup(self, session_id: str) -> Dict | None:
        """Is there a follow-up question we're still waiting on an answer to?
        Used so the NEXT user message resolves it (stores the answer as a
        memory via normal extraction, marks the original event completed)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM event_followups WHERE session_id = ? AND "
                "awaiting_answer = 1 AND resolved = 0 ORDER BY followup_at DESC LIMIT 1",
                (session_id,)).fetchone()
        return dict(row) if row else None

    def resolve_event_followup(self, fid: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE event_followups SET resolved = 1, awaiting_answer = 0 "
                "WHERE id = ?", (fid,))
            self._conn.commit()

    def mark_event_resolved_by_memory(self, memory_id: int) -> None:
        """An event was manually marked complete elsewhere (e.g. Settings ✓) —
        stop any pending encouragement/follow-up for it too."""
        with self._lock:
            self._conn.execute(
                "UPDATE event_followups SET resolved = 1, awaiting_answer = 0 "
                "WHERE memory_id = ?", (memory_id,))
            self._conn.commit()

    # ------------------------------------------------------------------
    # Device presence (delivery routing: companion device first, else WhatsApp)
    # ------------------------------------------------------------------
    def heartbeat_device(self, device_id: str, kind: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO device_presence (device_id, kind, last_seen) VALUES (?, ?, ?) "
                "ON CONFLICT(device_id) DO UPDATE SET last_seen = excluded.last_seen, "
                "kind = excluded.kind", (device_id, kind, _utcnow()))
            self._conn.commit()

    def connected_device(self, within_seconds: int = 120) -> Dict | None:
        """Most-recently-seen tablet/iot device still within its heartbeat TTL,
        or None if nothing's connected (caller falls back to WhatsApp)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=within_seconds)).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM device_presence WHERE last_seen >= ? "
                "ORDER BY last_seen DESC LIMIT 1", (cutoff,)).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Mood journal (passive, local-model-only extraction — see app/journal.py)
    # ------------------------------------------------------------------
    def add_mood_entry(self, date: str, time: str | None, mood_label: str,
                       intensity: int, why: str, source_channel: str | None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO mood_journal (date, time, mood_label, intensity, why, "
                "source_channel, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (date, time, mood_label, intensity, why, source_channel, _utcnow()))
            self._conn.commit()
            return int(cur.lastrowid)

    def list_mood_entries(self, since_date: str | None = None,
                          mood_filter: str | None = None) -> List[Dict]:
        q = "SELECT * FROM mood_journal WHERE 1=1"
        params: list = []
        if since_date:
            q += " AND date >= ?"
            params.append(since_date)
        if mood_filter:
            q += " AND mood_label = ?"
            params.append(mood_filter)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY date DESC, time DESC", params).fetchall()
        return [dict(r) for r in rows]

    def update_mood_entry(self, entry_id: int, **fields) -> Dict | None:
        allowed = {"mood_label", "intensity", "why"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return self.get_mood_entry(entry_id)
        sets = ", ".join(f"{c} = ?" for c in cols) + ", edited = 1"
        with self._lock:
            self._conn.execute(
                f"UPDATE mood_journal SET {sets} WHERE id = ?",
                [fields[c] for c in cols] + [entry_id])
            self._conn.commit()
        return self.get_mood_entry(entry_id)

    def get_mood_entry(self, entry_id: int) -> Dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mood_journal WHERE id = ?", (entry_id,)).fetchone()
        return dict(row) if row else None

    def delete_mood_entry(self, entry_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM mood_journal WHERE id = ?", (entry_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def unsynced_mood_entries(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM mood_journal WHERE synced_to_sheets = 0 "
                "ORDER BY id LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def mark_mood_synced(self, entry_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE mood_journal SET synced_to_sheets = 1 WHERE id = ?", (entry_id,))
            self._conn.commit()

    def mood_entries_for_day(self, date: str) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM mood_journal WHERE date = ? ORDER BY time", (date,)
            ).fetchall()
        return [dict(r) for r in rows]

    def backup_to(self, dest_path: Path) -> None:
        """Consistent online snapshot via SQLite's backup API (safe while the
        app is writing — unlike copying the file, which can catch a torn page)."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(dest_path))
            try:
                self._conn.backup(dest)
            finally:
                dest.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
