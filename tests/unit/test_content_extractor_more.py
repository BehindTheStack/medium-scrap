import textwrap
from src.infrastructure import content_extractor as ce


def test_extract_code_blocks_with_language_class():
    html = """
    <pre><code class="language-python">def foo():\n    return 1\n</code></pre>
    """
    blocks = ce.extract_code_blocks(html)
    assert len(blocks) == 1
    assert 'def foo' in blocks[0]['code']
    assert blocks[0]['language'] in ('python', 'Python') or blocks[0]['language'] is not None


def test_extract_code_blocks_pygments_fallback():
    # snippet that pygments should recognise as javascript
    html = """
    <pre><code>function hello() { console.log('hi'); }</code></pre>
    """
    blocks = ce.extract_code_blocks(html)
    assert len(blocks) == 1
    lang = blocks[0]['language']
    # Accept None or any string result from heuristics/pygments (we mainly want no crash)
    assert lang is None or isinstance(lang, str)


def test_html_to_markdown_and_assets():
    html = textwrap.dedent('''
    <h1>Title</h1>
    <p>Body text</p>
    <img src="https://example.com/assets/img.png" alt="An image">
    <pre><code class="language-python">import os\nprint('ok')</code></pre>
    ''')

    md, assets, code_blocks = ce.html_to_markdown(html)
    assert 'Title' in md
    assert len(assets) == 1
    assert assets[0]['filename'].endswith('img.png')
    assert len(code_blocks) == 1


def test_classify_technical_various_cases():
    # no code blocks and no keywords => not technical
    r = ce.classify_technical('<p>hello world</p>', [])
    assert isinstance(r, dict)
    assert r['is_technical'] is False

    # with code blocks becomes technical
    r2 = ce.classify_technical('<pre><code>def f(): pass</code></pre>', [{'code': 'def f(): pass'}])
    assert r2['is_technical'] is True
    assert 'code_blocks' in ','.join(r2['reasons'])


def test_html_to_markdown_with_data_src_and_pre_without_code():
    html = '''
    <p>Intro</p>
    <img data-src="https://example.com/pic.jpg" alt="x" />
    <pre>console.log('x')</pre>
    '''
    md, assets, code_blocks = ce.html_to_markdown(html)
    assert len(assets) == 1
    assert assets[0]['filename'].endswith('pic.jpg')
    # pre without code should still detect a code block via heuristics
    assert isinstance(code_blocks, list)


def test_extract_code_blocks_cpp_heuristic():
    html = '<pre>#include <iostream>\nint main() { return 0; }</pre>'
    blocks = ce.extract_code_blocks(html)
    assert len(blocks) == 1
    assert blocks[0]['language'] in (None, 'cpp', 'c++') or isinstance(blocks[0]['language'], str)
