from src.infrastructure import content_extractor


def test_html_to_markdown_and_code_extraction():
    html = '''
    <h1>Title</h1>
    <p>Some text</p>
    <img src="http://example.com/a.png" alt="A" />
    <pre><code class="language-python">print('hello')</code></pre>
    '''

    md, assets, code_blocks = content_extractor.html_to_markdown(html)

    assert 'print' in md or "print('hello')" in md
    assert len(assets) == 1
    assert any('a.png' in a['filename'] or 'a.png' in a.get('src', '') for a in assets)
    assert isinstance(code_blocks, list)
    assert any(cb.get('language') and 'python' in cb.get('language') for cb in code_blocks)

    cls = content_extractor.classify_technical(html, code_blocks)
    assert cls['is_technical'] is True
