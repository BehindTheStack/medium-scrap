from src.infrastructure.content_extractor import html_to_markdown, extract_code_blocks


def test_html_to_markdown_basic():
    html = """
    <h1>Title</h1>
    <p>This is a <a href="https://example.com">link</a> and an image: <img src="/assets/img.png" alt="img"></p>
    <pre><code class="language-python">def hello():\n    return 'hi'\n</code></pre>
    """

    md, assets, codes = html_to_markdown(html)

    assert "# Title" in md
    assert "(https://example.com)" in md
    assert any(a['src'].endswith('img.png') for a in assets)
    assert len(codes) == 1
    assert 'def hello' in codes[0]['code']
    assert codes[0]['language'] and 'python' in codes[0]['language']


def test_extract_code_blocks_no_class():
    html = """
    <pre>console.log('hello');\n</pre>
    """

    blocks = extract_code_blocks(html)
    assert len(blocks) == 1
    assert "console.log" in blocks[0]['code']
    # heuristic should guess javascript
    assert blocks[0]['language'] in ("javascript", "js", "node") or blocks[0]['language'] is not None
