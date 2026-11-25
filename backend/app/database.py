"""Local SQLite persistence for registered gym users, face embeddings, and workout history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "gym_trainer.db"
PHOTOS_DIR = DATA_DIR / "user_photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                fitness_level TEXT NOT NULL,
                photo_path TEXT,
                face_embedding TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                exercise TEXT NOT NULL,
                mode TEXT NOT NULL,
                total_reps INTEGER NOT NULL,
                correct_reps INTEGER NOT NULL,
                incorrect_reps INTEGER NOT NULL,
                form_score REAL NOT NULL,
                cues_json TEXT NOT NULL,
                video_url TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            """
        )


def create_user(
    name: str,
    age: int,
    height_cm: float,
    weight_kg: float,
    fitness_level: str,
    face_embedding: list[float] | list[list[float]],
    photo_path: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if face_embedding and isinstance(face_embedding[0], list):
        emb_payload: Any = [[float(v) for v in row] for row in face_embedding]  # type: ignore[union-attr]
    else:
        emb_payload = [float(v) for v in face_embedding]  # type: ignore[arg-type]
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO users (name, age, height_cm, weight_kg, fitness_level, photo_path, face_embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name.strip(),
                int(age),
                float(height_cm),
                float(weight_kg),
                fitness_level.strip().title(),
                photo_path,
                json.dumps(emb_payload),
                now,
            ),
        )
        user_id = cur.lastrowid
    return get_user(int(user_id))  # type: ignore[return-value]


def _row_to_user(row: sqlite3.Row, include_embedding: bool = False) -> dict[str, Any]:
    data = {
        "id": row["id"],
        "name": row["name"],
        "age": row["age"],
        "height_cm": row["height_cm"],
        "weight_kg": row["weight_kg"],
        "fitness_level": row["fitness_level"],
        "photo_url": f"/static/photos/{Path(row['photo_path']).name}" if row["photo_path"] else None,
        "created_at": row["created_at"],
    }
    if include_embedding:
        data["face_embedding"] = json.loads(row["face_embedding"])
    return data


def list_users(include_embedding: bool = False) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    return [_row_to_user(r, include_embedding=include_embedding) for r in rows]


def get_user(user_id: int, include_embedding: bool = False) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row, include_embedding=include_embedding) if row else None


def delete_user(user_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return cur.rowcount > 0


def save_workout_session(
    user_id: int | None,
    exercise: str,
    mode: str,
    total_reps: int,
    correct_reps: int,
    incorrect_reps: int,
    form_score: float,
    cues: list[str],
    video_url: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO workout_sessions
                (user_id, exercise, mode, total_reps, correct_reps, incorrect_reps, form_score, cues_json, video_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                exercise,
                mode,
                int(total_reps),
                int(correct_reps),
                int(incorrect_reps),
                float(round(form_score, 2)),
                json.dumps(cues),
                video_url,
                now,
            ),
        )
        session_id = cur.lastrowid

        # Dynamically update user's fitness level based on rolling recent form score if >= 3 sessions
        if user_id is not None:
            recent = conn.execute(
                "SELECT form_score FROM workout_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 5",
                (user_id,),
            ).fetchall()
            if len(recent) >= 3:
                avg_score = sum(r["form_score"] for r in recent) / len(recent)
                new_level = "Advanced" if avg_score >= 85 else ("Intermediate" if avg_score >= 65 else "Beginner")
                conn.execute("UPDATE users SET fitness_level = ? WHERE id = ?", (new_level, user_id))

    return {
        "session_id": session_id,
        "user_id": user_id,
        "exercise": exercise,
        "mode": mode,
        "total_reps": total_reps,
        "correct_reps": correct_reps,
        "incorrect_reps": incorrect_reps,
        "form_score": round(form_score, 2),
        "cues": cues,
        "video_url": video_url,
        "created_at": now,
    }


def list_user_sessions(user_id: int) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM workout_sessions WHERE user_id = ? ORDER BY id DESC LIMIT 30",
            (user_id,),
        ).fetchall()
    return [
        {
            "session_id": r["id"],
            "user_id": r["user_id"],
            "exercise": r["exercise"],
            "mode": r["mode"],
            "total_reps": r["total_reps"],
            "correct_reps": r["correct_reps"],
            "incorrect_reps": r["incorrect_reps"],
            "form_score": r["form_score"],
            "cues": json.loads(r["cues_json"]),
            "video_url": r["video_url"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
