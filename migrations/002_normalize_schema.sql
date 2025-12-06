-- Migration 002: Normalize and Enhance Schema
-- Creates new optimized tables and migrates data

-- Step 1: Create new normalized posts table
CREATE TABLE posts_new (
    -- Core identity
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    publication TEXT NOT NULL,
    
    -- Basic metadata
    title TEXT NOT NULL,
    subtitle TEXT,
    url TEXT NOT NULL,
    slug TEXT,  -- Extract from URL for better SEO
    
    -- Author info (normalized structure)
    author_id TEXT,
    author_name TEXT,
    author_username TEXT,
    
    -- Publishing info
    published_at INTEGER,  -- Unix timestamp for better querying
    updated_at INTEGER,    -- When post was last updated on Medium
    reading_time INTEGER DEFAULT 0,
    
    -- Engagement metrics
    claps INTEGER DEFAULT 0,
    responses INTEGER DEFAULT 0,  -- NEW: comment count
    
    -- Content storage
    content_html TEXT,
    content_markdown TEXT,
    content_text TEXT,  -- For full-text search
    content_length INTEGER,  -- Character count for stats
    
    -- Media
    featured_image_url TEXT,  -- NEW: Hero image
    
    -- Categorization
    tags TEXT,  -- JSON array
    topics TEXT,  -- NEW: Medium topics (JSON array)
    
    -- Technical analysis
    is_technical BOOLEAN DEFAULT 0,
    technical_score REAL DEFAULT 0.0,
    code_blocks INTEGER DEFAULT 0,
    code_languages TEXT,  -- NEW: JSON array of detected languages
    
    -- Collection metadata
    collection_mode TEXT DEFAULT 'metadata',  -- metadata, full, enriched
    collected_at INTEGER NOT NULL,
    last_synced_at INTEGER,
    
    -- Full metadata dump
    metadata_json TEXT,  -- Complete raw metadata
    
    -- Timestamps (Unix for performance)
    created_at INTEGER NOT NULL,
    updated_locally_at INTEGER NOT NULL,
    
    UNIQUE(url)  -- Prevent duplicates by URL
);

-- Step 2: Create ML discovery table (separate from posts)
CREATE TABLE ml_discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    
    -- ML version tracking
    model_version TEXT NOT NULL,  -- e.g., "modern-v1", "legacy-v1"
    pipeline_type TEXT NOT NULL,  -- "legacy", "modern", "modern-llm"
    processed_at INTEGER NOT NULL,
    
    -- Layer/Topic clustering
    layers TEXT,  -- JSON array: ["Backend", "Database"]
    
    -- Technology extraction
    tech_stack TEXT,  -- JSON array of {name, type, confidence}
    
    -- Pattern detection
    patterns TEXT,  -- JSON array of {pattern, confidence, source}
    
    -- Solution/Problem analysis
    solutions TEXT,  -- JSON array of solution descriptions
    problem TEXT,  -- Main problem addressed
    approach TEXT,  -- Main technical approach
    
    -- Embeddings
    embedding_model TEXT,  -- e.g., "all-MiniLM-L6-v2"
    embedding_vector BLOB,  -- Binary vector for semantic search
    
    -- Quality metrics
    extraction_confidence REAL,  -- Overall confidence score
    requires_review BOOLEAN DEFAULT 0,  -- Flag for manual review
    
    FOREIGN KEY(post_id) REFERENCES posts_new(id) ON DELETE CASCADE,
    UNIQUE(post_id, model_version)  -- One discovery per model version
);

-- Step 3: Create authors table (normalized)
CREATE TABLE authors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    username TEXT UNIQUE,
    bio TEXT,
    followers_count INTEGER DEFAULT 0,
    avatar_url TEXT,
    
    -- Stats
    posts_count INTEGER DEFAULT 0,
    technical_posts_count INTEGER DEFAULT 0,
    
    -- Timestamps
    first_seen_at INTEGER NOT NULL,
    last_updated_at INTEGER NOT NULL
);

-- Step 4: Create processing history table
CREATE TABLE processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    processing_type TEXT NOT NULL,  -- collection, ml_legacy, ml_modern, ml_llm, timeline
    status TEXT NOT NULL,  -- success, failed, skipped
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    duration_ms INTEGER,
    error_message TEXT,
    metadata TEXT,  -- JSON with additional context
    
    FOREIGN KEY(post_id) REFERENCES posts_new(id) ON DELETE CASCADE
);

-- Step 5: Create full-text search virtual table
CREATE VIRTUAL TABLE posts_fts USING fts5(
    post_id UNINDEXED,
    title,
    content_text,
    author_name,
    tags,
    content=posts_new,
    content_rowid=rowid
);

-- Step 6: Migrate data from old posts table
INSERT INTO posts_new (
    id, source, publication, title, url, slug,
    author_id, author_name, author_username,
    published_at, reading_time, claps,
    content_html, content_markdown, content_text, content_length,
    tags, is_technical, technical_score, code_blocks,
    collection_mode, collected_at, last_synced_at,
    metadata_json, created_at, updated_locally_at
)
SELECT 
    id,
    source,
    publication,
    COALESCE(title, 'Untitled'),
    COALESCE(url, 'https://medium.com/unknown'),
    -- Slug (will be extracted properly in application code)
    id,  -- Use ID as slug for now
    -- Author fields (normalized)
    NULL,  -- author_id (will be populated later)
    author,
    LOWER(REPLACE(COALESCE(author, 'unknown'), ' ', '_')),
    -- Convert ISO timestamps to Unix
    CASE 
        WHEN published_at IS NOT NULL THEN strftime('%s', published_at)
        ELSE NULL
    END,
    reading_time,
    claps,
    content_html,
    content_markdown,
    content_text,
    LENGTH(COALESCE(content_text, '')),
    tags,
    COALESCE(is_technical, 0),
    COALESCE(technical_score, 0.0),
    COALESCE(code_blocks, 0),
    COALESCE(collection_mode, 'metadata'),
    strftime('%s', collected_at),
    CASE 
        WHEN last_updated IS NOT NULL THEN strftime('%s', last_updated)
        ELSE NULL
    END,
    metadata_json,
    strftime('%s', collected_at),
    strftime('%s', COALESCE(last_updated, collected_at))
FROM posts
WHERE id IS NOT NULL AND source IS NOT NULL AND publication IS NOT NULL;

-- Step 7: Migrate ML data to new table (only if ML classified)
INSERT INTO ml_discoveries (
    post_id, model_version, pipeline_type, processed_at,
    layers, tech_stack, patterns, solutions, problem, approach,
    extraction_confidence
)
SELECT 
    id,
    'legacy-v1',  -- Mark as legacy extraction
    'legacy',
    COALESCE(strftime('%s', ml_processed_at), strftime('%s', 'now')),
    ml_layers,
    tech_stack,
    patterns,
    solutions,
    problem,
    approach,
    0.5  -- Default confidence for legacy data
FROM posts
WHERE ml_classified = 1 AND id IN (SELECT id FROM posts_new);

-- Step 8: Extract and normalize authors
INSERT OR IGNORE INTO authors (
    id, name, username, posts_count, first_seen_at, last_updated_at
)
SELECT 
    author_username,
    author_name,
    author_username,
    COUNT(*),
    MIN(published_at),
    MAX(updated_locally_at)
FROM posts_new
WHERE author_name IS NOT NULL
GROUP BY author_username;

-- Step 9: Update posts with author_id
UPDATE posts_new
SET author_id = (
    SELECT id FROM authors WHERE authors.username = posts_new.author_username
);

-- Step 10: Create indexes on new tables
CREATE INDEX idx_posts_new_source ON posts_new(source);
CREATE INDEX idx_posts_new_publication ON posts_new(publication);
CREATE INDEX idx_posts_new_author_id ON posts_new(author_id);
CREATE INDEX idx_posts_new_published_at ON posts_new(published_at);
CREATE INDEX idx_posts_new_collected_at ON posts_new(collected_at);
CREATE INDEX idx_posts_new_is_technical ON posts_new(is_technical);
CREATE INDEX idx_posts_new_technical_score ON posts_new(technical_score);
CREATE INDEX idx_posts_new_has_content ON posts_new(content_markdown) WHERE content_markdown IS NOT NULL;

CREATE INDEX idx_ml_post_id ON ml_discoveries(post_id);
CREATE INDEX idx_ml_model_version ON ml_discoveries(model_version);
CREATE INDEX idx_ml_pipeline_type ON ml_discoveries(pipeline_type);
CREATE INDEX idx_ml_processed_at ON ml_discoveries(processed_at);

CREATE INDEX idx_authors_username ON authors(username);
CREATE INDEX idx_authors_name ON authors(name);

CREATE INDEX idx_proc_hist_post_id ON processing_history(post_id);
CREATE INDEX idx_proc_hist_type ON processing_history(processing_type);
CREATE INDEX idx_proc_hist_status ON processing_history(processing_type, status);

-- Step 11: Drop old posts table and rename new one
DROP TABLE posts;
ALTER TABLE posts_new RENAME TO posts;

-- Step 12: Create triggers for FTS
CREATE TRIGGER posts_fts_insert AFTER INSERT ON posts BEGIN
    INSERT INTO posts_fts(post_id, title, content_text, author_name, tags)
    VALUES (new.id, new.title, new.content_text, new.author_name, new.tags);
END;

CREATE TRIGGER posts_fts_delete AFTER DELETE ON posts BEGIN
    DELETE FROM posts_fts WHERE post_id = old.id;
END;

CREATE TRIGGER posts_fts_update AFTER UPDATE ON posts BEGIN
    DELETE FROM posts_fts WHERE post_id = old.id;
    INSERT INTO posts_fts(post_id, title, content_text, author_name, tags)
    VALUES (new.id, new.title, new.content_text, new.author_name, new.tags);
END;

-- Step 13: Populate FTS table with existing data
INSERT INTO posts_fts(post_id, title, content_text, author_name, tags)
SELECT id, title, content_text, author_name, tags FROM posts;
