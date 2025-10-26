"""Lightweight HTML -> Markdown converter and code extractor.

This module purposely avoids adding heavy new dependencies. It uses
the stdlib's HTMLParser + Pygments (already in the environment) for
language guessing when possible.

Public functions:
- html_to_markdown(html: str) -> tuple[str, list[dict], list[dict]]
    Returns (markdown, assets, code_blocks)
- extract_code_blocks(html: str) -> list[dict]
    Returns list of {'code': str, 'language': Optional[str]}
"""
from __future__ import annotations

import re
import html as _html
from html.parser import HTMLParser
from typing import List, Optional, Tuple

from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound


def _heuristic_guess_language(code: str) -> Optional[str]:
    """Simple heuristics to guess code language when Pygments fails."""
    sample = code[:200]
    if re.search(r"\bdef\s+\w+\(|\bimport\s+\w+", sample):
        return "python"
    if re.search(r"console\.log\(|\bfunction\s*\(|\bconst\s+\w+", sample):
        return "javascript"
    if re.search(r"#include\s+<|printf\(|scanf\(|std::", sample):
        return "c++"
    if re.search(r"public\s+static\s+void|System\.out\.println|class\s+\w+\{", sample):
        return "java"
    if re.search(r"^\s*SELECT\s+.+FROM\s+", sample, re.I):
        return "sql"
    return None


def extract_code_blocks(html: str) -> List[dict]:
    """Extract <pre><code> (and <pre>) blocks from HTML and detect languages.

    Returns list of {'code': str, 'language': Optional[str]}
    """
    blocks: List[dict] = []

    # Try to find <pre><code class="language-xxx">...</code></pre>
    pattern = re.compile(
        r"<pre[^>]*>\s*(?:<code(?P<attrs>[^>]*)>)?(?P<code>.*?)(?:</code>)?\s*</pre>",
        re.IGNORECASE | re.DOTALL,
    )

    for m in pattern.finditer(html):
        code_html = m.group('code') or ''
        attrs = m.group('attrs') or ''
        # strip HTML tags inside code block conservatively
        code_text = re.sub(r"<[^>]+>", '', code_html)
        code_text = _html.unescape(code_text)
        # detect language from class attr e.g. class="language-python"
        lang = None
        cls_match = re.search(r'class\s*=\s*"([^"]+)"', attrs)
        if cls_match:
            cls = cls_match.group(1)
            m2 = re.search(r'language-(\w+)', cls)
            if m2:
                lang = m2.group(1)

        # If no explicit language, try Pygments
        if not lang:
            try:
                lexer = guess_lexer(code_text)
                lang = lexer.name.lower()
            except ClassNotFound:
                lang = _heuristic_guess_language(code_text)

        blocks.append({'code': code_text.strip('\n'), 'language': lang})

    return blocks


class _MinimalHTMLToMarkdown(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._in_pre = False
        self._in_code = False
        self._code_buffer: List[str] = []
        self.assets: List[dict] = []
        self._link_href_stack: List[Optional[str]] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            level = int(tag[1])
            self.parts.append('\n' + ('#' * level) + ' ')
        elif tag == 'p':
            self.parts.append('\n\n')
        elif tag == 'br':
            self.parts.append('  \n')
        elif tag == 'pre':
            self._in_pre = True
            self._code_buffer = []
        elif tag == 'code':
            self._in_code = True
        elif tag == 'a':
            href = attrs.get('href')
            self._link_href_stack.append(href)
            # start link text; actual formatting happens in handle_endtag
        elif tag == 'img':
            src = attrs.get('src')
            alt = attrs.get('alt', '')
            if src:
                # Save asset metadata; actual download happens elsewhere
                self.assets.append({'src': src, 'alt': alt})
                filename = src.split('/')[-1]
                self.parts.append(f'![{alt}]({filename})')

    def handle_endtag(self, tag):
        if tag == 'pre':
            # flush code buffer as fenced block
            code = ''.join(self._code_buffer).rstrip('\n')
            self.parts.append('\n\n```\n')
            self.parts.append(code)
            self.parts.append('\n```\n')
            self._in_pre = False
            self._code_buffer = []
        elif tag == 'code':
            self._in_code = False
        elif tag == 'a':
            href = None
            if self._link_href_stack:
                href = self._link_href_stack.pop()
            # We don't have the link text separated here; keep it simple
            if href:
                self.parts.append(f' ({href})')

    def handle_data(self, data):
        if self._in_pre or self._in_code:
            self._code_buffer.append(data)
        else:
            # collapse multiple spaces
            text = data.replace('\n', ' ')
            self.parts.append(text)

    def get_markdown(self) -> Tuple[str, List[dict]]:
        md = ''.join(self.parts)
        # basic cleanup
        md = re.sub(r'\s+\n', '\n', md)
        md = md.strip() + '\n'
        return md, self.assets


def html_to_markdown(html: str) -> Tuple[str, List[dict], List[dict]]:
    """Convert HTML to a simple Markdown, extract assets and code blocks.

    Returns (markdown, assets, code_blocks)
    """
    parser = _MinimalHTMLToMarkdown()
    parser.feed(html)
    md, assets = parser.get_markdown()
    code_blocks = extract_code_blocks(html)
    return md, assets, code_blocks


__all__ = ["html_to_markdown", "extract_code_blocks"]
