#!/usr/bin/env python3
"""
Validate markdown files with YAML frontmatter
Usage: python scripts/validate_markdown.py [directory]
"""
import sys
import yaml
from pathlib import Path

def validate_markdown_file(file_path):
    """Validate a single markdown file"""
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        return False, f"Can't read file: {e}"
    
    # Check for frontmatter
    if not content.startswith('---\n'):
        return False, "No frontmatter found (should start with ---)"
    
    parts = content.split('---\n', 2)
    if len(parts) < 3:
        return False, "Invalid frontmatter structure"
    
    # Validate YAML
    try:
        frontmatter = yaml.safe_load(parts[1])
        if not isinstance(frontmatter, dict):
            return False, "Frontmatter is not a dict"
    except Exception as e:
        return False, f"Invalid YAML: {e}"
    
    # Check required fields
    required = ['id', 'title', 'author']
    missing = [f for f in required if f not in frontmatter]
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Check content exists
    md_content = parts[2].strip()
    if len(md_content) < 50:
        return False, "Content too short (< 50 chars)"
    
    return True, frontmatter

def main():
    if len(sys.argv) > 1:
        search_dir = Path(sys.argv[1])
    else:
        search_dir = Path('outputs')
    
    if not search_dir.exists():
        print(f"❌ Directory not found: {search_dir}")
        return 1
    
    md_files = list(search_dir.rglob('*.md'))
    
    if not md_files:
        print(f"❌ No markdown files found in {search_dir}")
        return 1
    
    print(f"📝 Validating {len(md_files)} markdown files...")
    print()
    
    valid = 0
    invalid = 0
    
    for md_file in md_files:
        is_valid, result = validate_markdown_file(md_file)
        
        if is_valid:
            valid += 1
            print(f"✅ {md_file.name}")
            if isinstance(result, dict):
                print(f"   Title: {result.get('title', 'N/A')[:60]}")
                print(f"   Author: {result.get('author', 'N/A')}")
                print(f"   Technical: {result.get('is_technical', 'N/A')} (score: {result.get('technical_score', 'N/A')})")
        else:
            invalid += 1
            print(f"❌ {md_file.name}")
            print(f"   Error: {result}")
        
        print()
    
    print("─" * 60)
    print(f"Summary: {valid} valid, {invalid} invalid (total: {len(md_files)})")
    print()
    
    if invalid > 0:
        print("⚠️  Some files need re-scraping with new format:")
        print("   uv run python main.py pipeline")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
