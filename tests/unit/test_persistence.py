import json
from types import SimpleNamespace


from src.infrastructure.persistence import persist_markdown_and_metadata


def make_post(id_value='abcd1234abcd', title='Test Post', slug='test-post', author_name='Author'):
    post = SimpleNamespace()
    post.id = SimpleNamespace(value=id_value)
    post.title = title
    post.slug = slug
    post.author = SimpleNamespace(name=author_name)
    return post


def test_persist_markdown_and_metadata(tmp_path):
    post = make_post()
    markdown = '# Hello\nThis is a test'
    assets = [{'src': 'https://example.com/images/logo.png', 'alt': 'logo'}]
    code_blocks = [{'code': "print('hi')", 'language': 'python'}]
    classifier = {'is_technical': True, 'score': 0.9, 'reasons': ['code_blocks:1']}

    out = persist_markdown_and_metadata(post, markdown, assets, str(tmp_path), code_blocks=code_blocks, classifier=classifier)

    # Files exist
    md_path = tmp_path / ('abcd1234abcd_Test_Post.md')
    assert md_path.exists()

    meta_path = tmp_path / ('abcd1234abcd_Test_Post.json')
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert meta['id'] == 'abcd1234abcd'
    assert isinstance(meta.get('code_blocks'), list)
    assert meta.get('classifier', {}).get('is_technical') is True
