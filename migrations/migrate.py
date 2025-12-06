"""Database Migration System.

Manages schema migrations with version tracking and rollback support.
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations"""
    
    def __init__(self, db_path: str, migrations_dir: str = "migrations"):
        self.db_path = Path(db_path)
        self.migrations_dir = Path(migrations_dir)
        self._ensure_migration_table()
    
    def _ensure_migration_table(self):
        """Create migrations tracking table if not exists"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    applied_at INTEGER NOT NULL,
                    execution_time_ms INTEGER,
                    checksum TEXT,
                    status TEXT DEFAULT 'applied'
                )
            """)
            conn.commit()
        finally:
            conn.close()
    
    def get_current_version(self) -> Optional[str]:
        """Get the latest applied migration version"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT version FROM schema_migrations 
                WHERE status = 'applied'
                ORDER BY version DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of all applied migrations"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT version FROM schema_migrations 
                WHERE status = 'applied'
                ORDER BY version
            """)
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_pending_migrations(self) -> List[Path]:
        """Get list of migrations that haven't been applied"""
        applied = set(self.get_applied_migrations())
        all_migrations = sorted(self.migrations_dir.glob("*.sql"))
        
        pending = []
        for migration_file in all_migrations:
            version = migration_file.stem.split('_')[0]
            if version not in applied:
                pending.append(migration_file)
        
        return pending
    
    def apply_migration(self, migration_file: Path, dry_run: bool = False) -> bool:
        """Apply a single migration file
        
        Args:
            migration_file: Path to migration SQL file
            dry_run: If True, only validate without applying
            
        Returns:
            True if successful, False otherwise
        """
        version = migration_file.stem.split('_')[0]
        name = '_'.join(migration_file.stem.split('_')[1:])
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Applying migration {version}: {name}")
        
        # Read migration SQL
        sql = migration_file.read_text()
        
        if dry_run:
            logger.info(f"Would execute {len(sql)} characters of SQL")
            return True
        
        # Execute migration
        conn = sqlite3.connect(self.db_path)
        start_time = datetime.now()
        
        try:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Execute migration in a transaction
            conn.executescript(sql)
            conn.commit()
            
            # Record migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            conn.execute("""
                INSERT INTO schema_migrations (version, name, applied_at, execution_time_ms, status)
                VALUES (?, ?, ?, ?, 'applied')
            """, (version, name, int(datetime.now().timestamp()), execution_time))
            conn.commit()
            
            logger.info(f"✓ Migration {version} applied successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            conn.rollback()
            logger.error(f"✗ Migration {version} failed: {e}")
            
            # Record failed migration
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO schema_migrations 
                    (version, name, applied_at, status)
                    VALUES (?, ?, ?, 'failed')
                """, (version, name, int(datetime.now().timestamp())))
                conn.commit()
            except:
                pass
            
            return False
        finally:
            conn.close()
    
    def migrate_up(self, target_version: Optional[str] = None, dry_run: bool = False) -> bool:
        """Apply all pending migrations up to target version
        
        Args:
            target_version: Stop at this version (None = apply all)
            dry_run: If True, only validate without applying
            
        Returns:
            True if all migrations successful, False otherwise
        """
        pending = self.get_pending_migrations()
        
        if not pending:
            logger.info("No pending migrations")
            return True
        
        logger.info(f"Found {len(pending)} pending migrations")
        
        for migration_file in pending:
            version = migration_file.stem.split('_')[0]
            
            # Stop if we've reached target version
            if target_version and version > target_version:
                break
            
            success = self.apply_migration(migration_file, dry_run=dry_run)
            if not success:
                logger.error(f"Migration failed, stopping at {version}")
                return False
        
        return True
    
    def create_backup(self) -> Path:
        """Create a backup of the database before migration
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_{timestamp}.db"
        
        # Use SQLite backup API
        source = sqlite3.connect(self.db_path)
        dest = sqlite3.connect(backup_path)
        
        try:
            source.backup(dest)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        finally:
            source.close()
            dest.close()
    
    def get_migration_status(self) -> dict:
        """Get detailed migration status
        
        Returns:
            Dictionary with migration information
        """
        current = self.get_current_version()
        applied = self.get_applied_migrations()
        pending = self.get_pending_migrations()
        
        return {
            'current_version': current,
            'applied_count': len(applied),
            'pending_count': len(pending),
            'applied_migrations': applied,
            'pending_migrations': [f.stem for f in pending]
        }


def main():
    """CLI for running migrations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('command', choices=['status', 'up', 'backup'],
                       help='Migration command')
    parser.add_argument('--db', default='outputs/pipeline.db',
                       help='Database path')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate without applying')
    parser.add_argument('--target', help='Target version')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = MigrationManager(args.db)
    
    if args.command == 'status':
        status = manager.get_migration_status()
        print(f"\n{'='*60}")
        print(f"Database Migration Status")
        print(f"{'='*60}")
        print(f"Current Version: {status['current_version'] or 'None'}")
        print(f"Applied: {status['applied_count']} migrations")
        print(f"Pending: {status['pending_count']} migrations")
        
        if status['applied_migrations']:
            print(f"\nApplied Migrations:")
            for version in status['applied_migrations']:
                print(f"  ✓ {version}")
        
        if status['pending_migrations']:
            print(f"\nPending Migrations:")
            for migration in status['pending_migrations']:
                print(f"  • {migration}")
        print()
    
    elif args.command == 'backup':
        backup_path = manager.create_backup()
        print(f"✓ Backup created: {backup_path}")
    
    elif args.command == 'up':
        print("\nRunning migrations...")
        if not args.dry_run:
            # Create backup first
            manager.create_backup()
        
        success = manager.migrate_up(
            target_version=args.target,
            dry_run=args.dry_run
        )
        
        if success:
            print(f"\n✓ All migrations {'validated' if args.dry_run else 'applied'} successfully")
        else:
            print(f"\n✗ Migration failed")
            return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
