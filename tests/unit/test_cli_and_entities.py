import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from src.presentation.cli import PostFormatter, CLIController
from src.domain.entities.publication import (
    PostId, Author, Post, PublicationId, PublicationConfig, PublicationType
)
from src.application.use_cases.scrape_posts import ScrapePostsResponse


def make_sample_post():
    pid = PostId('aaaaaaaaaaaa')
    author = Author(id='auth1', name='Author One', username='authone')
    published = datetime.utcnow() - timedelta(days=2)
    post = Post(
        id=pid,
        title='Test Post',
        slug='test-post',
        author=author,
        published_at=published,
        reading_time=5.0,
        subtitle='Sub'
    )
    return post


def test_post_to_dict_and_recent_update():
    post = make_sample_post()
    d = post.to_dict()
    assert d['id'] == 'aaaaaaaaaaaa'
    assert d['title'] == 'Test Post'
    assert 'published_at' in d
    assert not post.is_recently_updated


def test_post_formatter_json_and_ids(tmp_path):
    posts = [make_sample_post()]
    fmt = PostFormatter()

    json_str = fmt.format_as_json(posts)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert parsed[0]['id'] == 'aaaaaaaaaaaa'

    ids_str = fmt.format_as_ids(posts)
    assert 'aaaaaaaaaaaa' in ids_str

    # Table formatting should not raise
    fmt.format_as_table(posts, publication_name='TestPub')


def test_cli_controller_handle_success_and_file_write(tmp_path, monkeypatch):
    # Create a fake publication config
    pub_config = PublicationConfig(
        id=PublicationId('testpub'),
        name='testpub',
        type=PublicationType.MEDIUM_HOSTED,
        domain='medium.com/testpub',
        graphql_url='https://medium.com/graphql',
        known_post_ids=[]
    )

    post = make_sample_post()
    response = ScrapePostsResponse(
        posts=[post],
        publication_config=pub_config,
        total_posts_found=1,
        discovery_method='test'
    )

    # Fake use case that returns our response
    class FakeUseCase:
        def execute(self, request):
            return response

    controller = CLIController(FakeUseCase())

    # Monkeypatch _save_to_file to write to tmp_path
    out_file = tmp_path / "out.json"

    # Ensure save_to_file works via _handle_successful_response when format=json and output_file provided
    controller._handle_successful_response(response, format_type='json', output_file=str(out_file))
    assert out_file.exists()
    content = out_file.read_text(encoding='utf-8')
    assert 'aaaaaaaaaaaa' in content


def test_handle_failed_response_prints(monkeypatch, capsys):
    class FakeUseCase:
        pass

    controller = CLIController(FakeUseCase())
    # Create a minimal ScrapePostsResponse-like object
    fake_resp = MagicMock()
    fake_resp.total_posts_found = 0

    controller._handle_failed_response(fake_resp)
    captured = capsys.readouterr()
    assert 'No posts found' in captured.out or 'No posts found' in captured.err
