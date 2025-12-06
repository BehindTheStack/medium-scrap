-- Migration 001: Base Schema Creation
-- This represents the current state (already applied via _init_db)
-- Kept for reference only - do not run on existing database

-- Posts table - Main content storage
CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    publication TEXT NOT NULL,
    title TEXT,
    author TEXT,
    url TEXT,
    published_at TEXT,
    reading_time INTEGER,
    claps INTEGER,
    tags TEXT,
    
    content_html TEXT,
    content_markdown TEXT,
    content_text TEXT,
    metadata_json TEXT,
    
    collected_at TEXT NOT NULL,
    last_updated TEXT,
    collection_mode TEXT,
    
    has_markdown BOOLEAN DEFAULT 0,
    has_json BOOLEAN DEFAULT 0,
    markdown_path TEXT,
    json_path TEXT,
    
    is_technical BOOLEAN,
    technical_score REAL,
    code_blocks INTEGER DEFAULT 0,
    
    in_timeline BOOLEAN DEFAULT 0,
    in_ml_training BOOLEAN DEFAULT 0,
    ml_classified BOOLEAN DEFAULT 0,
    ml_layers TEXT,
    
    timeline_processed_at TEXT,
    ml_processed_at TEXT,
    
    -- ML Discovery fields (added via ALTER)
    tech_stack TEXT,
    patterns TEXT,
    solutions TEXT,
    problem TEXT,
    approach TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source);
CREATE INDEX IF NOT EXISTS idx_posts_publication ON posts(publication);
CREATE INDEX IF NOT EXISTS idx_posts_collected_at ON posts(collected_at);
CREATE INDEX IF NOT EXISTS idx_posts_has_markdown ON posts(has_markdown);
CREATE INDEX IF NOT EXISTS idx_posts_ml_classified ON posts(ml_classified);
CREATE INDEX IF NOT EXISTS idx_posts_is_technical ON posts(is_technical);
CREATE INDEX IF NOT EXISTS idx_posts_has_content ON posts(content_markdown);

-- Supporting tables
CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT,
    mode TEXT,
    posts_collected INTEGER DEFAULT 0,
    posts_new INTEGER DEFAULT 0,
    posts_updated INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS timeline_builds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication TEXT NOT NULL,
    built_at TEXT NOT NULL,
    post_count INTEGER,
    output_file TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS ml_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT,
    model_path TEXT,
    training_posts INTEGER,
    classified_posts INTEGER,
    accuracy REAL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT
);
