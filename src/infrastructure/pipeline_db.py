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


class PipelineDB:
    """Database for tracking pipeline state and caching results"""
    
    def __init__(self, db_path: str = "outputs/pipeline.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._migrate_db()  # Run migrations
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        try:
            # Improve concurrency for multiple writers/readers
            # Enable Write-Ahead Logging (WAL) and relaxed synchronous for speed
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
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
    
    def _migrate_db(self):
        """Apply database migrations"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if content columns exist
            cursor.execute("PRAGMA table_info(posts)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add new content columns if they don't exist
            if 'content_html' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN content_html TEXT")
            if 'content_markdown' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN content_markdown TEXT")
            if 'content_text' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN content_text TEXT")
            if 'metadata_json' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN metadata_json TEXT")
            
            # Add ML discovery columns (NEW)
            if 'tech_stack' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN tech_stack TEXT")  # JSON array
            if 'patterns' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN patterns TEXT")  # JSON array
            if 'solutions' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN solutions TEXT")  # JSON array
            if 'problem' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN problem TEXT")
            if 'approach' not in columns:
                cursor.execute("ALTER TABLE posts ADD COLUMN approach TEXT")
            
            # Create index for content search (only after column exists)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_has_content 
                ON posts(content_markdown)
            """)
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Posts table - tracks all collected posts
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
    
    def post_exists(self, post_id: str) -> bool:
        """Check if post already exists in database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM posts WHERE id = ?", (post_id,))
            return cursor.fetchone() is not None
    
    def get_post(self, post_id: str) -> Optional[Dict]:
        """Get post by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def add_or_update_post(self, post_data: Dict[str, Any]) -> None:
        """Add new post or update existing one"""
        now = datetime.utcnow().isoformat()
        
        # Extract author name if it's a dict
        author = post_data.get('author')
        if isinstance(author, dict):
            author = author.get('name', author.get('username', ''))
        
        # Convert lists/dicts to JSON
        tags = json.dumps(post_data.get('tags', []))
        ml_layers = json.dumps(post_data.get('ml_layers', []))
        metadata_json = json.dumps(post_data.get('metadata', {})) if post_data.get('metadata') else None
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            exists = self.post_exists(post_data['id'])
            
            if exists:
                # Update (including source and publication to fix legacy data)
                cursor.execute("""
                    UPDATE posts SET
                        source = ?,
                        publication = ?,
                        title = ?,
                        author = ?,
                        url = ?,
                        published_at = ?,
                        reading_time = ?,
                        claps = ?,
                        tags = ?,
                        content_html = ?,
                        content_markdown = ?,
                        content_text = ?,
                        metadata_json = ?,
                        last_updated = ?,
                        collection_mode = ?,
                        has_markdown = ?,
                        has_json = ?,
                        markdown_path = ?,
                        json_path = ?,
                        is_technical = ?,
                        technical_score = ?,
                        code_blocks = ?,
                        in_timeline = ?,
                        ml_classified = ?,
                        ml_layers = ?
                    WHERE id = ?
                """, (
                    post_data['source'],
                    post_data['publication'],
                    post_data.get('title'),
                    author,
                    post_data.get('url'),
                    post_data.get('published_at'),
                    post_data.get('reading_time'),
                    post_data.get('claps'),
                    tags,
                    post_data.get('content_html'),
                    post_data.get('content_markdown'),
                    post_data.get('content_text'),
                    metadata_json,
                    now,
                    post_data.get('collection_mode'),
                    post_data.get('has_markdown', False),
                    post_data.get('has_json', False),
                    post_data.get('markdown_path'),
                    post_data.get('json_path'),
                    post_data.get('is_technical'),
                    post_data.get('technical_score'),
                    post_data.get('code_blocks', 0),
                    post_data.get('in_timeline', False),
                    post_data.get('ml_classified', False),
                    ml_layers,
                    post_data['id']
                ))
            else:
                # Insert
                cursor.execute("""
                    INSERT INTO posts (
                        id, source, publication, title, author, url,
                        published_at, reading_time, claps, tags,
                        content_html, content_markdown, content_text, metadata_json,
                        collected_at, last_updated, collection_mode,
                        has_markdown, has_json, markdown_path, json_path,
                        is_technical, technical_score, code_blocks,
                        in_timeline, ml_classified, ml_layers
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post_data['id'],
                    post_data['source'],
                    post_data['publication'],
                    post_data.get('title'),
                    author,
                    post_data.get('url'),
                    post_data.get('published_at'),
                    post_data.get('reading_time'),
                    post_data.get('claps'),
                    tags,
                    post_data.get('content_html'),
                    post_data.get('content_markdown'),
                    post_data.get('content_text'),
                    metadata_json,
                    now,
                    now,
                    post_data.get('collection_mode', 'metadata'),
                    post_data.get('has_markdown', False),
                    post_data.get('has_json', False),
                    post_data.get('markdown_path'),
                    post_data.get('json_path'),
                    post_data.get('is_technical'),
                    post_data.get('technical_score'),
                    post_data.get('code_blocks', 0),
                    post_data.get('in_timeline', False),
                    post_data.get('ml_classified', False),
                    ml_layers
                ))
    
    def get_posts_by_source(self, source: str, with_markdown_only: bool = False) -> List[Dict]:
        """Get all posts for a source"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if with_markdown_only:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? AND has_markdown = 1
                    ORDER BY published_at DESC
                """, (source,))
            else:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? 
                    ORDER BY published_at DESC
                """, (source,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_posts_needing_markdown(self, source: Optional[str] = None) -> List[Dict]:
        """Get posts that need markdown extraction"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if source:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE source = ? AND has_markdown = 0
                """, (source,))
            else:
                cursor.execute("""
                    SELECT * FROM posts 
                    WHERE has_markdown = 0
                """)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_ml_discovery(self, post_id: str, ml_data: Dict[str, Any]) -> None:
        """
        Update ML discovery fields for a post.
        
        Args:
            post_id: Post ID
            ml_data: Dictionary containing:
                - layers: List[str] - Topic labels from clustering
                - tech_stack: List[Dict] - Technologies extracted by NER
                - patterns: List[Dict] - Architectural patterns from zero-shot
                - solutions: List[str] - Solution descriptions
                - problem: str (optional) - Main problem addressed
                - approach: str (optional) - Main approach taken
        """
        now = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize JSON fields
            layers_json = json.dumps(ml_data.get('layers', []))
            tech_stack_json = json.dumps(ml_data.get('tech_stack', []))
            patterns_json = json.dumps(ml_data.get('patterns', []))
            solutions_json = json.dumps(ml_data.get('solutions', []))
            
            cursor.execute("""
                UPDATE posts
                SET 
                    ml_layers = ?,
                    tech_stack = ?,
                    patterns = ?,
                    solutions = ?,
                    problem = ?,
                    approach = ?,
                    ml_classified = 1,
                    ml_processed_at = ?
                WHERE id = ?
            """, (
                layers_json,
                tech_stack_json,
                patterns_json,
                solutions_json,
                ml_data.get('problem'),
                ml_data.get('approach'),
                now,
                post_id
            ))
    
    def get_posts_needing_ml_classification(self) -> List[Dict]:
        """Get posts with markdown but not ML classified"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM posts 
                WHERE has_markdown = 1 AND ml_classified = 0
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def mark_posts_in_timeline(self, post_ids: List[str], publication: str) -> None:
        """Mark posts as included in timeline"""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                UPDATE posts 
                SET in_timeline = 1, timeline_processed_at = ?
                WHERE id = ?
            """, [(now, pid) for pid in post_ids])
    
    def mark_posts_ml_classified(self, post_ids: List[str], layers_map: Dict[str, List[str]]) -> None:
        """Mark posts as ML classified with their layers"""
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for post_id in post_ids:
                layers = json.dumps(layers_map.get(post_id, []))
                cursor.execute("""
                    UPDATE posts 
                    SET ml_classified = 1, ml_processed_at = ?, ml_layers = ?
                    WHERE id = ?
                """, (now, layers, post_id))
    
    # =========================================================================
    # Collection Runs
    # =========================================================================
    
    def start_collection_run(self, source: str, mode: str) -> int:
        """Start a new collection run, return run ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO collection_runs (source, started_at, status, mode)
                VALUES (?, ?, 'running', ?)
            """, (source, datetime.utcnow().isoformat(), mode))
            return cursor.lastrowid
    
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
            
            # Posts with markdown
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE has_markdown = 1")
            with_markdown = cursor.fetchone()['count']
            
            # Posts in timeline
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE in_timeline = 1")
            in_timeline = cursor.fetchone()['count']
            
            # ML classified
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE ml_classified = 1")
            ml_classified = cursor.fetchone()['count']
            
            # Technical posts
            cursor.execute("SELECT COUNT(*) as count FROM posts WHERE is_technical = 1")
            technical = cursor.fetchone()['count']
            
            return {
                'total_posts': total,
                'by_source': by_source,
                'with_markdown': with_markdown,
                'in_timeline': in_timeline,
                'ml_classified': ml_classified,
                'technical_posts': technical,
                'needs_markdown': total - with_markdown,
                'needs_ml': with_markdown - ml_classified
            }
    
    # =========================================================================
    # Cache
    # =========================================================================
    
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
    
    def cache_delete(self, key: str) -> None:
        """Delete cache entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
    
    # =========================================================================
    # Content Retrieval Methods
    # =========================================================================
    
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
    
    def get_posts_needing_ml(self, source: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict]:
        """Get posts that need ML processing (have content but not ML classified)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM posts 
                WHERE content_markdown IS NOT NULL
                AND LENGTH(content_markdown) > 100
                AND (ml_classified IS NULL OR ml_classified = 0)
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
    
    def search_content(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Simple full-text search in post content
        
        Args:
            query: Search term
            limit: Max results
        
        Returns:
            List of matching posts with highlights
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Simple LIKE search (could be improved with FTS5)
            cursor.execute("""
                SELECT id, source, title, author, url, published_at,
                       SUBSTR(content_text, 1, 200) as snippet
                FROM posts 
                WHERE content_text LIKE ? OR title LIKE ?
                ORDER BY published_at DESC
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit))
            
            return [dict(row) for row in cursor.fetchall()]


__all__ = ['PipelineDB']
