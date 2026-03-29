#!/usr/bin/env python3
"""load_data.py — Initialize SQLite database and load cell-count.csv."""

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("cell_counts.db")
CSV_PATH = Path("cell-count.csv")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    condition TEXT NOT NULL,
    sex TEXT NOT NULL CHECK (sex IN ('M', 'F', 'other'))
);

CREATE TABLE IF NOT EXISTS enrollments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id TEXT NOT NULL REFERENCES subjects(id),
    project_id TEXT NOT NULL REFERENCES projects(id),
    treatment TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age > 0),
    UNIQUE (subject_id, project_id)
);

CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    enrollment_id INTEGER NOT NULL REFERENCES enrollments(id),
    sample_type TEXT NOT NULL,
    time_from_treatment_start INTEGER NOT NULL CHECK (time_from_treatment_start >= 0),
    response TEXT CHECK (response IN ('yes', 'no') OR response IS NULL),
    b_cell INTEGER NOT NULL,
    cd8_t_cell INTEGER NOT NULL,
    cd4_t_cell INTEGER NOT NULL,
    nk_cell INTEGER NOT NULL,
    monocyte INTEGER NOT NULL
);

-- View that flattens the normalized schema for analytics queries.
-- All pipeline and dashboard code reads from this view, so the
-- normalization is invisible to consumers.
CREATE VIEW IF NOT EXISTS sample_view AS
SELECT
    sa.id          AS sample,
    sub.id         AS subject,
    e.project_id   AS project,
    sub.condition,
    e.age,
    sub.sex,
    e.treatment,
    sa.sample_type,
    sa.time_from_treatment_start,
    sa.response,
    sa.b_cell,
    sa.cd8_t_cell,
    sa.cd4_t_cell,
    sa.nk_cell,
    sa.monocyte
FROM samples sa
JOIN enrollments e ON sa.enrollment_id = e.id
JOIN subjects sub ON e.subject_id = sub.id;


"""


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)

    projects = set()
    subjects = {}
    enrollments = {}  # (subject_id, project_id) -> treatment
    samples = []

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            projects.add(row["project"])

            subj = row["subject"]
            if subj not in subjects:
                subjects[subj] = (
                    row["condition"],
                    row["sex"],
                )

            enroll_key = (subj, row["project"])
            if enroll_key not in enrollments:
                enrollments[enroll_key] = (row["treatment"], int(row["age"]))

            response = row["response"] if row["response"] else None
            samples.append((
                row["sample"],
                enroll_key,
                row["sample_type"],
                int(row["time_from_treatment_start"]),
                response,
                int(row["b_cell"]),
                int(row["cd8_t_cell"]),
                int(row["cd4_t_cell"]),
                int(row["nk_cell"]),
                int(row["monocyte"]),
            ))

    conn.executemany(
        "INSERT INTO projects (id) VALUES (?)",
        [(pid,) for pid in sorted(projects)],
    )

    conn.executemany(
        "INSERT INTO subjects (id, condition, sex) VALUES (?, ?, ?)",
        [(sid, *vals) for sid, vals in subjects.items()],
    )

    conn.executemany(
        "INSERT INTO enrollments (subject_id, project_id, treatment, age) "
        "VALUES (?, ?, ?, ?)",
        [(subj, proj, treat, age) for (subj, proj), (treat, age) in enrollments.items()],
    )

    # Build enrollment ID lookup for sample insertion
    enrollment_ids = {}
    for row in conn.execute("SELECT id, subject_id, project_id FROM enrollments"):
        enrollment_ids[(row[1], row[2])] = row[0]

    conn.executemany(
        "INSERT INTO samples (id, enrollment_id, sample_type, time_from_treatment_start, "
        "response, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (s[0], enrollment_ids[s[1]], *s[2:])
            for s in samples
        ],
    )

    conn.commit()

    cur = conn.cursor()
    proj_count = cur.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    subj_count = cur.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
    enrl_count = cur.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
    samp_count = cur.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    print(f"Database created: {DB_PATH}")
    print(f"  projects:    {proj_count}")
    print(f"  subjects:    {subj_count}")
    print(f"  enrollments: {enrl_count}")
    print(f"  samples:     {samp_count}")

    conn.close()


if __name__ == "__main__":
    main()
