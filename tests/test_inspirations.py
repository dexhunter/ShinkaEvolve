"""Tests for inspiration context selection (shinka/database/inspirations.py)."""

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from shinka.database import DatabaseConfig, Program, ProgramDatabase
from shinka.database.inspirations import (
    CombinedContextSelector,
    TopKInspirationSelector,
)


def _make_program(
    program_id: str, score: float, parent_id: Optional[str] = None
) -> Program:
    return Program(
        id=program_id,
        code=f"def f():\n    return {score}\n",
        parent_id=parent_id,
        combined_score=score,
        public_metrics={"fitness": score},
        correct=True,
        complexity=1.0,
        embedding=[0.1, 0.2, 0.3],
        metadata={"job": program_id},
    )


def _open_db(db_path: Path, num_islands: int = 1) -> ProgramDatabase:
    config = DatabaseConfig(
        db_path=str(db_path),
        num_islands=num_islands,
        archive_size=20,
        elite_selection_ratio=1.0,
        num_archive_inspirations=2,
        num_top_k_inspirations=3,
        enforce_island_separation=True,
        migration_rate=0.0,
        enable_dynamic_islands=False,
    )
    return ProgramDatabase(config=config, embedding_model="")


def _get(db: ProgramDatabase, program_id: str) -> Program:
    program = db.get(program_id)
    assert program is not None, program_id
    return program


def _make_topk_selector(
    db: ProgramDatabase, **overrides: Any
) -> TopKInspirationSelector:
    kwargs = dict(
        cursor=db.cursor,
        conn=db.conn,
        config=db.config,
        get_program_func=db.get,
        best_program_id=db.best_program_id,
        get_island_idx_func=db.island_manager.get_island_idx,
        program_from_row_func=db._program_from_row,
    )
    kwargs.update(overrides)
    return TopKInspirationSelector(**kwargs)


def _add_ranked_programs(db: ProgramDatabase, count: int = 12) -> None:
    for i in range(count):
        db.add(_make_program(f"prog_{i:02d}", float(i)))


def test_topk_selector_returns_best_archived_programs_in_score_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _open_db(Path(tmpdir) / "test.db")
        try:
            _add_ranked_programs(db)
            parent = _get(db, "prog_11")
            excluded = [_get(db, "prog_10")]

            top_k = _make_topk_selector(db).sample_context(parent, excluded, 3)

            assert [p.id for p in top_k] == ["prog_09", "prog_08", "prog_07"]
            # Full program objects, identical to a direct lookup.
            for program in top_k:
                assert program.to_dict() == _get(db, program.id).to_dict()
        finally:
            db.close()


def test_topk_selector_falls_back_to_public_metrics_mean():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _open_db(Path(tmpdir) / "test.db")
        try:
            _add_ranked_programs(db)
            # No combined_score: rank by the mean of public_metrics, and
            # programs without any usable metrics sort last.
            db.cursor.execute(
                "UPDATE programs SET combined_score = NULL, public_metrics = ? "
                "WHERE id = ?",
                (json.dumps({"a": 100.0, "b": 50.0}), "prog_05"),
            )
            db.cursor.execute(
                "UPDATE programs SET combined_score = NULL, public_metrics = ? "
                "WHERE id = ?",
                (json.dumps({}), "prog_04"),
            )
            db.cursor.execute(
                "UPDATE programs SET combined_score = NULL, public_metrics = NULL "
                "WHERE id = ?",
                ("prog_03",),
            )
            db.conn.commit()
            parent = _get(db, "prog_11")
            excluded = [_get(db, "prog_10")]
            selector = _make_topk_selector(db)

            top_k = selector.sample_context(parent, excluded, 3)
            assert [p.id for p in top_k] == ["prog_05", "prog_09", "prog_08"]

            everything = selector.sample_context(parent, excluded, 20)
            assert len(everything) == 10
            assert {p.id for p in everything[-2:]} == {"prog_03", "prog_04"}
        finally:
            db.close()


def test_topk_selector_loads_only_k_full_programs():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _open_db(Path(tmpdir) / "test.db")
        try:
            _add_ranked_programs(db)
            loaded: List[str] = []

            def counting_get(program_id: str) -> Optional[Program]:
                loaded.append(program_id)
                return db.get(program_id)

            def counting_from_row(row: sqlite3.Row) -> Optional[Program]:
                loaded.append(row["id"])
                return db._program_from_row(row)

            selector = _make_topk_selector(
                db,
                get_program_func=counting_get,
                program_from_row_func=counting_from_row,
            )
            parent = _get(db, "prog_11")

            top_k = selector.sample_context(parent, [], 3)

            assert [p.id for p in top_k] == ["prog_10", "prog_09", "prog_08"]
            # Only the k winners are hydrated, not the whole archive.
            assert loaded == [p.id for p in top_k]
        finally:
            db.close()


def test_topk_selector_respects_island_separation():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _open_db(Path(tmpdir) / "test.db", num_islands=2)
        try:
            db.add(_make_program("root_a", 1.0))
            db.add(_make_program("root_b", 2.0))
            for i in range(10):
                root = "root_a" if i % 2 == 0 else "root_b"
                db.add(_make_program(f"child_{i:02d}", 10.0 + i, parent_id=root))
            parent = _get(db, "child_00")

            top_k = _make_topk_selector(db).sample_context(parent, [], 3)

            db.cursor.execute("SELECT program_id FROM archive")
            archived = [_get(db, row["program_id"]) for row in db.cursor.fetchall()]
            same_island = [
                p
                for p in archived
                if p.island_idx == parent.island_idx and p.id != parent.id
            ]
            same_island.sort(key=lambda p: p.combined_score, reverse=True)
            assert [p.id for p in top_k] == [p.id for p in same_island[:3]]
            assert all(p.island_idx == parent.island_idx for p in top_k)
        finally:
            db.close()


def test_combined_context_selector_excludes_archive_inspirations_from_topk():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _open_db(Path(tmpdir) / "test.db")
        try:
            _add_ranked_programs(db)
            assert db.best_program_id == "prog_11"
            parent = _get(db, "prog_00")
            selector = CombinedContextSelector(
                cursor=db.cursor,
                conn=db.conn,
                config=db.config,
                get_program_func=db.get,
                best_program_id=db.best_program_id,
                get_island_idx_func=db.island_manager.get_island_idx,
                program_from_row_func=db._program_from_row,
            )

            archive_insp, top_k = selector.sample_context(parent, 2, 3)

            assert [p.id for p in archive_insp] == ["prog_11", "prog_10"]
            assert [p.id for p in top_k] == ["prog_09", "prog_08", "prog_07"]
        finally:
            db.close()
