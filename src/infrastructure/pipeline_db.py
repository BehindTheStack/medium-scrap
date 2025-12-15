"""
SQLite database for caching and tracking pipeline state.
Optimizes collection, processing, and avoids re-processing.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

def with_sqlite_retry(
    max_retries: int = 8,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
):
    """
    Decorator para retry automático em erros 'database is locked' do SQLite.
    Usa backoff exponencial + jitter.
    """
    
    import time
    import random
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except sqlite3.OperationalError as e:
                    msg = str(e).lower()

                    if any(x in msg for x in ["database is locked", "database is busy", "unable to open database file"]):
                        if attempt >= max_retries - 1:
                            raise

                        sleep_time = min(delay, max_delay)
                        sleep_time += random.uniform(0, sleep_time * 0.1)

                        time.sleep(sleep_time)
                        delay *= 2
                        continue

                    raise

        return wrapper

    return decorator



class PipelineDB:
    """Database for tracking pipeline state and caching results"""
    
    def __init__(self, db_path: str = "outputs/pipeline.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()  # Ensure schema exists (migration-aware)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)  # Increased timeout to 30s
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        try:
            # Improve concurrency for multiple writers/readers
            # Enable Write-Ahead Logging (WAL) and relaxed synchronous for speed
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")  # 30s busy timeout
        except Exception:
            # If PRAGMA fails, continue with default settings
            pass
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _ensure_schema(self):
        """Ensure database schema exists (migration-aware)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if schema_migrations table exists (migration 002 indicator)
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='schema_migrations'
            """)
            has_migrations = cursor.fetchone() is not None
            
            if not has_migrations:
                # Legacy database - run old init
                self._init_db_legacy()
            else:
                # New schema from migration 002
                # Just ensure supporting tables exist
                self._ensure_supporting_tables()
    
    def _ensure_supporting_tables(self):
        """Ensure supporting tables exist (collection_runs, timeline_builds, etc)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # These tables are not affected by migration 002
            cursor.execute("""
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
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_builds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    publication TEXT NOT NULL,
                    built_at TEXT NOT NULL,
                    post_count INTEGER,
                    output_file TEXT,
                    status TEXT
                )
            """)
            
            cursor.execute("""
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
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
    
    def _init_db_legacy(self):
        """Initialize legacy database schema (pre-migration)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Legacy Posts table - will be migrated by migration 002
            cursor.execute("""
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
                    tags TEXT,  -- JSON array
                    
                    -- Full content storage (NEW)
                    content_html TEXT,  -- Original HTML content
                    content_markdown TEXT,  -- Converted markdown
                    content_text TEXT,  -- Plain text (for search)
                    metadata_json TEXT,  -- Full post metadata as JSON
                    
                    -- Collection metadata
                    collected_at TEXT NOT NULL,
                    last_updated TEXT,
                    collection_mode TEXT,  -- metadata, full, technical
                    
                    -- Content availability (DEPRECATED but kept for compatibility)
                    has_markdown BOOLEAN DEFAULT 0,
                    has_json BOOLEAN DEFAULT 0,
                    markdown_path TEXT,
                    json_path TEXT,
                    
                    -- Technical classification
                    is_technical BOOLEAN,
                    technical_score REAL,
                    code_blocks INTEGER DEFAULT 0,
                    
                    -- Processing status
                    in_timeline BOOLEAN DEFAULT 0,
                    in_ml_training BOOLEAN DEFAULT 0,
                    ml_classified BOOLEAN DEFAULT 0,
                    ml_layers TEXT,  -- JSON array of architecture layers
                    
                    -- Timestamps
                    timeline_processed_at TEXT,
                    ml_processed_at TEXT
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_source 
                ON posts(source)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_publication 
                ON posts(publication)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_collected_at 
                ON posts(collected_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_has_markdown 
                ON posts(has_markdown)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_ml_classified 
                ON posts(ml_classified)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_is_technical 
                ON posts(is_technical)
            """)
            
            # Collection runs table - tracks pipeline executions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT,  -- running, completed, failed
                    mode TEXT,
                    posts_collected INTEGER DEFAULT 0,
                    posts_new INTEGER DEFAULT 0,
                    posts_updated INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)
            
            # Timeline builds table - tracks timeline generation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timeline_builds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    publication TEXT NOT NULL,
                    built_at TEXT NOT NULL,
                    post_count INTEGER,
                    output_file TEXT,
                    status TEXT  -- success, failed
                )
            """)
            
            # ML runs table - tracks ML training and classification
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_type TEXT NOT NULL,  -- training, classification
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT,  -- running, completed, failed
                    model_path TEXT,
                    training_posts INTEGER,
                    classified_posts INTEGER,
                    accuracy REAL,
                    error_message TEXT
                )
            """)
            
            # Cache table - generic key-value cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,  -- JSON
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
            """)
    
    # =========================================================================
    # Posts Management
    # =========================================================================
    
    @with_sqlite_retry()
    def post_exists(self, post_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM posts WHERE id = ?", (post_id,))
            return cur.fetchone() is not None
    
    @with_sqlite_retry()
    def get_post(self, post_id: str) -> Optional[Dict]:
        """Get post by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        
    @with_sqlite_retry()
    def add_or_update_post(self, post_data: Dict[str, Any]) -> None:
        """Add new post or update existing one"""
        now_unix = int(datetime.utcnow().timestamp())
        
        # Extract author info
        author_name = None
        author_username = None
        if isinstance(post_data.get('author'), dict):
            author_name = post_data['author'].get('name')
            author_username = post_data['author'].get('username')
        elif post_data.get('author'):
            author_name = post_data['author']
            author_username = author_name.lower().replace(' ', '_')
        
        # Convert published_at to Unix timestamp if it's ISO string
        published_at = post_data.get('published_at')
        if isinstance(published_at, str):
            try:
                dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                published_at = int(dt.timestamp())
            except:
                published_at = None
        
        # Convert lists/dicts to JSON
        tags = json.dumps(post_data.get('tags', []))
        topics = json.dumps(post_data.get('topics', []))
        code_languages = json.dumps(post_data.get('code_languages', []))
        metadata_json = json.dumps(post_data.get('metadata', {})) if post_data.get('metadata') else None
        
        # Calculate content length
        content_text = post_data.get('content_text', '')
        content_length = len(content_text) if content_text else 0
        
        # Extract slug from URL
        url = post_data.get('url', '')
        slug = None
        if url and '/' in url:
            slug = url.rstrip('/').split('/')[-1]
        if not slug:
            slug = post_data['id']
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            exists = self.post_exists(post_data['id'])
            
            if exists:
                # Update with new schema
                cursor.execute("""
                    UPDATE posts SET
                        source = ?,
                        publication = ?,
                        title = ?,
                        subtitle = ?,
                        url = ?,
                        slug = ?,
                        author_name = ?,
                        author_username = ?,
                        published_at = ?,
                        reading_time = ?,
                        claps = ?,
                        responses = ?,
                        tags = ?,
                        topics = ?,
                        content_html = ?,
                        content_markdown = ?,
                        content_text = ?,
                        content_length = ?,
                        featured_image_url = ?,
                        metadata_json = ?,
                        updated_locally_at = ?,
                        collection_mode = ?,
                        is_technical = ?,
                        technical_score = ?,
                        code_blocks = ?,
                        code_languages = ?
                    WHERE id = ?
                """, (
                    post_data['source'],
                    post_data['publication'],
                    post_data.get('title', 'Untitled'),
                    post_data.get('subtitle'),
                    url,
                    slug,
                    author_name,
                    author_username,
                    published_at,
                    post_data.get('reading_time', 0),
                    post_data.get('claps', 0),
                    post_data.get('responses', 0),
                    tags,
                    topics,
                    post_data.get('content_html'),
                    post_data.get('content_markdown'),
                    content_text,
                    content_length,
                    post_data.get('featured_image_url'),
                    metadata_json,
                    now_unix,
                    post_data.get('collection_mode', 'metadata'),
                    post_data.get('is_technical'),
                    post_data.get('technical_score'),
                    post_data.get('code_blocks', 0),
                    code_languages,
                    post_data['id']
                ))
            else:
                # Insert with new schema
                cursor.execute("""
                    INSERT INTO posts (
                        id, source, publication, title, subtitle, url, slug,
                        author_name, author_username,
                        published_at, reading_time, claps, responses,
                        tags, topics,
                        content_html, content_markdown, content_text, content_length,
                        featured_image_url, metadata_json,
                        collected_at, updated_locally_at, collection_mode,
                        is_technical, technical_score, code_blocks, code_languages,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_data['id'],
                    post_data['source'],
                    post_data['publication'],
                    post_data.get('title', 'Untitled'),
                    post_data.get('subtitle'),
                    url,
                    slug,
                    author_name,
                    author_username,
                    published_at,
                    post_data.get('reading_time', 0),
                    post_data.get('claps', 0),
                    post_data.get('responses', 0),
                    tags,
                    topics,
                    post_data.get('content_html'),
                    post_data.get('content_markdown'),
                    content_text,
                    content_length,
                    post_data.get('featured_image_url'),
                    metadata_json,
                    now_unix,
                    now_unix,
                    post_data.get('collection_mode', 'metadata'),
                    post_data.get('is_technical'),
                    post_data.get('technical_score'),
                    post_data.get('code_blocks', 0),
                    code_languages,
                    now_unix
                ))
            
            # Update or create author record
            if author_name:
                self._upsert_author(author_username or author_name, author_name)
    
    @with_sqlite_retry()
    def _upsert_author(self, author_id: str, name: str) -> None:
        """Insert or update author record"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now_unix = int(datetime.utcnow().timestamp())
            
            # Check if author exists
            cursor.execute("SELECT id FROM authors WHERE id = ?", (author_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                cursor.execute("""
                    UPDATE authors 
                    SET name = ?, last_updated_at = ?
                    WHERE id = ?
                """, (name, now_unix, author_id))
            else:
                username = author_id if '@' not in author_id else author_id.split('@')[0]
                cursor.execute("""
                    INSERT INTO authors (id, name, username, first_seen_at, last_updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (author_id, name, username, now_unix, now_unix))
    
    @with_sqlite_retry()
    def get_posts_by_source(self, source: str, with_markdown_only: bool = False) -> List[Dict]:
        """Get all posts for a source"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if with_markdown_only:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? AND content_markdown IS NOT NULL
                    ORDER BY published_at DESC
                """, (source,))
            else:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? 
                    ORDER BY published_at DESC
                """, (source,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry()
    def get_posts_needing_markdown(self, source: Optional[str] = None) -> List[Dict]:
        """Get posts that need markdown extraction"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if source:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? AND content_markdown IS NULL
                """, (source,))
            else:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE content_markdown IS NULL
                """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry(max_retries=12)
    def update_ml_discovery(self, post_id: str, ml_data: Dict[str, Any], model_version: str = 'legacy-v1', pipeline_type: str = 'legacy') -> None:
        """
        Save ML discovery data for a post in ml_discoveries table.
        
        Args:
            post_id: Post ID
            ml_data: Dictionary containing:
                - layers: List[str] - Topic labels from clustering
                - tech_stack: List[Dict] - Technologies extracted by NER
                - patterns: List[Dict] - Architectural patterns from zero-shot
                - solutions: List[str] - Solution descriptions
                - problem: str (optional) - Main problem addressed
                - approach: str (optional) - Main approach taken
                - embedding_vector: bytes (optional) - Vector for semantic search
                - extraction_confidence: float (optional) - Overall confidence
            model_version: Version identifier (e.g., 'modern-v1', 'legacy-v1')
            pipeline_type: Type of pipeline ('legacy', 'modern', 'modern-llm')
        """
        
        now_unix = int(datetime.utcnow().timestamp())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Serialize JSON fields (always convert to JSON string)
            layers_json = json.dumps(ml_data.get('layers', []))
            tech_stack_json = json.dumps(ml_data.get('tech_stack', []))
            patterns_json = json.dumps(ml_data.get('patterns', []))
            solutions_json = json.dumps(ml_data.get('solutions', []))
            # Sanitize scalar fields - ensure they're the correct type
            problem = ml_data.get('problem')
            if not isinstance(problem, (str, type(None))):
                problem = None
            approach = ml_data.get('approach')
            if not isinstance(approach, (str, type(None))):
                approach = None
            embedding_model = ml_data.get('embedding_model')
            if not isinstance(embedding_model, (str, type(None))):
                embedding_model = None
            embedding_vector = ml_data.get('embedding_vector')
            if not isinstance(embedding_vector, (bytes, type(None))):
                embedding_vector = None
            extraction_confidence = ml_data.get('extraction_confidence', 0.5)
            if not isinstance(extraction_confidence, (int, float)):
                extraction_confidence = 0.5
            # Check if discovery already exists for this version
            cursor.execute("""
                SELECT id FROM ml_discoveries 
                WHERE post_id = ? AND model_version = ?
            """, (post_id, model_version))
            exists = cursor.fetchone()
            if exists:
                # Update existing discovery
                cursor.execute("""
                    UPDATE ml_discoveries
                    SET 
                        pipeline_type = ?,
                        processed_at = ?,
                        layers = ?,
                        tech_stack = ?,
                        patterns = ?,
                        solutions = ?,
                        problem = ?,
                        approach = ?,
                        embedding_model = ?,
                        embedding_vector = ?,
                        extraction_confidence = ?
                    WHERE post_id = ? AND model_version = ?
                """, (
                    pipeline_type,
                    now_unix,
                    layers_json,
                    tech_stack_json,
                    patterns_json,
                    solutions_json,
                    problem,
                    approach,
                    embedding_model,
                    embedding_vector,
                    extraction_confidence,
                    post_id,
                    model_version
                ))
            else:
                # Insert new discovery
                cursor.execute("""
                    INSERT INTO ml_discoveries (
                        post_id, model_version, pipeline_type, processed_at,
                        layers, tech_stack, patterns, solutions, problem, approach,
                        embedding_model, embedding_vector, extraction_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_id,
                    model_version,
                    pipeline_type,
                    now_unix,
                    layers_json,
                    tech_stack_json,
                    patterns_json,
                    solutions_json,
                    problem,
                    approach,
                    embedding_model,
                    embedding_vector,
                    extraction_confidence
                ))
    
    @with_sqlite_retry()
    def get_posts_needing_ml_classification(self, model_version: str = 'legacy-v1') -> List[Dict]:
        """Get posts with content but not yet classified by specified model version"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.* FROM posts p
                LEFT JOIN ml_discoveries md ON p.id = md.post_id AND md.model_version = ?
                WHERE p.content_markdown IS NOT NULL 
                AND md.id IS NULL
            """, (model_version,))
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry()
    def get_ml_discovery(self, post_id: str, model_version: str = 'legacy-v1') -> Optional[Dict]:
        """Get ML discovery data for a specific post and model version"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ml_discoveries 
                WHERE post_id = ? AND model_version = ?
            """, (post_id, model_version))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # =========================================================================
    # Collection Runs
    # =========================================================================
    
    @with_sqlite_retry()
    def start_collection_run(self, source: str, mode: str) -> int:
        """Start a new collection run, return run ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collection_runs (source, started_at, status, mode)
                VALUES (?, ?, 'running', ?)
            """, (source, datetime.utcnow().isoformat(), mode))
            return cursor.lastrowid
    
    @with_sqlite_retry()
    def complete_collection_run(self, run_id: int, posts_collected: int, 
                               posts_new: int, posts_updated: int, 
                               status: str = 'completed', error: str = None) -> None:
        """Complete a collection run"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE collection_runs 
                SET completed_at = ?, status = ?, 
                    posts_collected = ?, posts_new = ?, posts_updated = ?,
                    error_message = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), status, posts_collected, 
                 posts_new, posts_updated, error, run_id))
    
    @with_sqlite_retry()
    def get_last_collection_run(self, source: str) -> Optional[Dict]:
        """Get last collection run for a source"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM collection_runs 
                WHERE source = ? 
                ORDER BY started_at DESC 
                LIMIT 1
            """, (source,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    @with_sqlite_retry()
    def get_stats(self) -> Dict:
        """Get overall statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Total posts
            cursor.execute("SELECT COUNT(*) as total FROM posts")
            total = cursor.fetchone()['total']
            
            # Posts by source
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM posts 
                GROUP BY source
            """)
            by_source = {row['source']: row['count'] for row in cursor.fetchall()}
            
            # Posts with markdown content
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE content_markdown IS NOT NULL")
            with_markdown = cursor.fetchone()['count']
            
            # Posts with ML discoveries
            cursor.execute("SELECT COUNT(DISTINCT post_id) as count FROM ml_discoveries")
            ml_classified = cursor.fetchone()['count']
            
            # Technical posts
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE is_technical = 1")
            technical = cursor.fetchone()['count']
            
            # Total authors
            cursor.execute("SELECT COUNT(*) as count FROM authors")
            total_authors = cursor.fetchone()['count']
            
            return {
                'total_posts': total,
                'by_source': by_source,
                'with_markdown': with_markdown,
                'ml_classified': ml_classified,
                'technical_posts': technical,
                'total_authors': total_authors,
                'needs_markdown': total - with_markdown,
                'needs_ml': with_markdown - ml_classified
            }
    
    # =========================================================================
    # Cache
    # =========================================================================
    
    @with_sqlite_retry()
    def cache_set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set cache value"""
        now = datetime.utcnow()
        expires_at = None
        if ttl_seconds:
            from datetime import timedelta
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (key, json.dumps(value), now.isoformat(), expires_at))
    
    @with_sqlite_retry()
    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value, expires_at FROM cache WHERE key = ?
            """, (key,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Check expiration
            if row['expires_at']:
                expires = datetime.fromisoformat(row['expires_at'])
                if datetime.utcnow() > expires:
                    # Expired, delete and return None
                    cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                    return None
            
            return json.loads(row['value'])
    
    @with_sqlite_retry()
    def cache_delete(self, key: str) -> None:
        """Delete cache entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
    
    # =========================================================================
    # Content Retrieval Methods
    # =========================================================================
    
    @with_sqlite_retry()
    def get_post_content(self, post_id: str, format: str = 'markdown') -> Optional[str]:
        """
        Get post content in specified format
        
        Args:
            post_id: Post ID
            format: 'markdown', 'html', or 'text'
        
        Returns:
            Content string or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if format == 'markdown':
                cursor.execute("SELECT content_markdown FROM posts WHERE id = ?", (post_id,))
            elif format == 'html':
                cursor.execute("SELECT content_html FROM posts WHERE id = ?", (post_id,))
            elif format == 'text':
                cursor.execute("SELECT content_text FROM posts WHERE id = ?", (post_id,))
            else:
                raise ValueError(f"Invalid format: {format}")
            
            row = cursor.fetchone()
            return row[0] if row else None
    
    @with_sqlite_retry()
    def get_posts_with_content(self, source: Optional[str] = None, 
                               limit: Optional[int] = None) -> List[Dict]:
        """Get posts that have markdown content stored"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM posts 
                WHERE content_markdown IS NOT NULL
            """
            params = []
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            query += " ORDER BY published_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry()
    def get_posts_needing_ml(self, source: Optional[str] = None,
                            limit: Optional[int] = None,
                            model_version: Optional[str] = None) -> List[Dict]:
        """Get posts that need ML processing (have content but no ML discovery)
        
        Args:
            source: Filter by source name
            limit: Limit number of results
            model_version: Filter by posts not processed by this model version
            
        Returns:
            List of post dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT p.* FROM posts p
                LEFT JOIN ml_discoveries md ON p.id = md.post_id
            """
            
            if model_version:
                query += " AND md.model_version = ?"
            
            query += """
                WHERE p.content_markdown IS NOT NULL
                AND LENGTH(p.content_markdown) > 100
                AND md.post_id IS NULL
            """
            
            params = []
            if model_version:
                params.append(model_version)
            
            if source:
                query += " AND p.source = ?"
                params.append(source)
            
            query += " ORDER BY p.published_at DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry()
    def search_content(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Full-text search in post content using FTS5
        
        Args:
            query: Search term (supports FTS5 syntax)
            limit: Max results
        
        Returns:
            List of matching posts with highlights
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Use FTS5 - post_id is stored directly in FTS table
            cursor.execute("""
                SELECT p.id, p.source, p.title, p.author_name, p.url, p.published_at,
                       SUBSTR(p.content_text, 1, 200) as snippet
                FROM posts_fts fts
                JOIN posts p ON fts.post_id = p.id
                WHERE posts_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # Authors Management
    # =========================================================================
    
    @with_sqlite_retry()
    def get_author(self, author_id: str) -> Optional[Dict]:
        """Get author by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM authors WHERE id = ?", (author_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    @with_sqlite_retry()
    def get_top_authors(self, limit: int = 10, technical_only: bool = False) -> List[Dict]:
        """Get top authors by post count"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            order_by = "technical_posts_count" if technical_only else "posts_count"
            cursor.execute(f"""
                SELECT * FROM authors 
                WHERE {order_by} > 0
                ORDER BY {order_by} DESC 
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    @with_sqlite_retry()
    def update_author_stats(self) -> None:
        """Recalculate author statistics from posts"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Update posts count
            cursor.execute("""
                UPDATE authors
                SET posts_count = (
                    SELECT COUNT(*) FROM posts 
                    WHERE posts.author_username = authors.username
                ),
                technical_posts_count = (
                    SELECT COUNT(*) FROM posts 
                    WHERE posts.author_username = authors.username 
                    AND posts.is_technical = 1
                )
            """)


__all__ = ['PipelineDB']
