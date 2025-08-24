"""
Infrastructure Adapters - External System Integration
Following Adapter Pattern and Dependency Inversion Principle
"""

import httpx
import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.entities.publication import (
    Post, PostId, PublicationConfig, PublicationType, Author, PublicationId
)
from ...domain.repositories.base import PostRepository, PublicationRepository, SessionRepository


class MediumApiAdapter(PostRepository):
    """
    Adapter for Medium GraphQL API
    Implements Repository interface for external API access
    """
    
    def __init__(self):
        self._base_headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
            "apollographql-client-name": "lite",
            "apollographql-client-version": "main-20250822-202538-95c3fbb66d",
            "medium-frontend-app": "lite/main-20250822-202538-95c3fbb66d",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
        }
    
    def get_posts_by_ids(self, post_ids: List[PostId], config: PublicationConfig) -> List[Post]:
        """Retrieve posts by their IDs using GraphQL API"""
        posts = []
        headers = self._get_headers_for_config(config)
        
        # Process in batches of 25 (API limit)
        batch_size = 25
        for i in range(0, len(post_ids), batch_size):
            batch = post_ids[i:i + batch_size]
            batch_posts = self._fetch_post_batch([pid.value for pid in batch], config, headers)
            posts.extend(batch_posts)
            
            # Rate limiting
            time.sleep(0.5)
        
        return posts
    
    def discover_post_ids(self, config: PublicationConfig, limit: Optional[int] = 25) -> List[PostId]:
        """Discover post IDs automatically from publication"""
        discovered_ids = []
        
        try:
            # Strategy 1: GraphQL Publication Query
            discovered_ids = self._discover_via_graphql(config, limit)
            if discovered_ids:
                return discovered_ids
        except Exception:
            pass
        
        try:
            # Strategy 2: HTML Scraping
            discovered_ids = self._discover_via_html_scraping(config, limit)
            if discovered_ids:
                return discovered_ids
        except Exception:
            pass
        
        return discovered_ids
    
    def get_posts_from_publication_feed(self, config: PublicationConfig, limit: Optional[int] = 25) -> List[Post]:
        """Get posts directly from publication feed"""
        # Implementation would use publication feed GraphQL query
        # For now, return empty list as fallback
        return []
    
    def _fetch_post_batch(self, post_id_strings: List[str], config: PublicationConfig, headers: Dict[str, str]) -> List[Post]:
        """Fetch a batch of posts from the API"""
        query = self._build_post_query(post_id_strings)
        
        try:
            with httpx.Client(verify=False, timeout=30.0) as client:
                response = client.post(config.graphql_url, headers=headers, json=query)
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            
            if "errors" in data or not data.get("data") or not data["data"].get("postResults"):
                return []
            
            return [self._parse_post(post_data) for post_data in data["data"]["postResults"] 
                    if post_data.get("__typename") == "Post"]
            
        except Exception:
            return []
    
    def _discover_via_graphql(self, config: PublicationConfig, limit: Optional[int]) -> List[PostId]:
        """Discover post IDs via GraphQL publication query with automatic pagination"""
        headers = self._get_headers_for_config(config)
        all_post_ids = []
        cursor = None
        page_size = 25  # GraphQL limit is usually 25 per page
        collected = 0
        
        # If no limit, set a high number to continue until no more pages
        max_to_collect = limit if limit is not None else float('inf')
        
        while collected < max_to_collect:
            remaining = max_to_collect - collected if limit is not None else page_size
            current_page_size = min(page_size, int(remaining)) if limit is not None else page_size
            
            query = self._build_publication_query(config.id.value, current_page_size, cursor)
            
            try:
                with httpx.Client(verify=False, timeout=30.0) as client:
                    response = client.post(config.graphql_url, headers=headers, json=query)
                
                if response.status_code != 200:
                    break
                    
                data = response.json()
                if (data.get("errors") or not data.get("data") or 
                    not data["data"].get("publication") or 
                    not data["data"]["publication"].get("posts")):
                    break
                
                posts_data = data["data"]["publication"]["posts"]
                edges = posts_data.get("edges", [])
                
                if not edges:
                    break
                
                # Collect post IDs from this page
                page_ids = [PostId(edge["node"]["id"]) for edge in edges if edge.get("node")]
                all_post_ids.extend(page_ids)
                collected += len(page_ids)
                
                # Check if there are more pages
                page_info = posts_data.get("pageInfo", {})
                if not page_info.get("hasNextPage"):
                    break
                    
                cursor = page_info.get("endCursor")
                if not cursor:
                    break
                    
            except Exception:
                break
        
        return all_post_ids
    
    def _discover_via_html_scraping(self, config: PublicationConfig, limit: Optional[int]) -> List[PostId]:
        """Discover post IDs via HTML scraping with pagination support"""
        all_ids = set()
        page = 0
        
        # If no limit, set a high number
        max_to_collect = limit if limit is not None else float('inf')
        
        while len(all_ids) < max_to_collect:
            if config.is_custom_domain:
                # For custom domains, try different page formats
                if page == 0:
                    url = f"https://{config.domain}"
                else:
                    url = f"https://{config.domain}/?page={page}"
            else:
                # For Medium-hosted publications
                if page == 0:
                    url = f"https://medium.com/{config.id.value}"
                else:
                    url = f"https://medium.com/{config.id.value}?page={page}"
            
            try:
                with httpx.Client(verify=False, timeout=30.0, follow_redirects=True) as client:
                    response = client.get(url, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    })
                
                if response.status_code != 200:
                    break
                
                html_content = response.text
                
                # Extract post IDs using regex
                id_pattern = r'[a-f0-9]{12}'
                potential_ids = re.findall(id_pattern, html_content)
                
                # Filter unique IDs
                page_ids = set(potential_ids)
                
                # If no new IDs found, stop pagination
                if not page_ids or page_ids.issubset(all_ids):
                    break
                
                all_ids.update(page_ids)
                page += 1
                
                # Add delay between requests
                if page > 1:
                    time.sleep(1.0)
                    
                # Safety limit to avoid infinite loops (especially for --all)
                if page > 50:  # Increased for --all support
                    break
                    
            except Exception:
                break
        
        # Convert to list and limit if specified
        unique_ids = list(all_ids)
        if limit is not None:
            limited_ids = unique_ids[:limit]
        else:
            limited_ids = unique_ids
        
        return [PostId(id_str) for id_str in limited_ids]
    
    def _get_headers_for_config(self, config: PublicationConfig) -> Dict[str, str]:
        """Get appropriate headers for publication configuration"""
        headers = self._base_headers.copy()
        
        if config.is_custom_domain:
            headers.update({
                "origin": f"https://{config.domain}",
                "referer": f"https://{config.domain}/",
            })
        else:
            headers.update({
                "origin": "https://medium.com",
                "referer": "https://medium.com/",
            })
        
        return headers
    
    def _build_post_query(self, post_ids: List[str]) -> Dict[str, Any]:
        """Build GraphQL query for fetching posts by IDs"""
        return {
            "operationName": "PublicationSectionPostsQuery",
            "variables": {"postIds": post_ids},
            "query": """query PublicationSectionPostsQuery($postIds: [ID!]!) {
                postResults(postIds: $postIds) {
                    ... on Post {
                        id
                        title
                        uniqueSlug
                        firstPublishedAt
                        latestPublishedAt
                        readingTime
                        creator {
                            id
                            name
                            username
                        }
                        extendedPreviewContent {
                            subtitle
                        }
                        __typename
                    }
                    __typename
                }
            }"""
        }
    
    def _build_publication_query(self, publication_id: str, limit: int, cursor: str = None) -> Dict[str, Any]:
        """Build GraphQL query for discovering posts from publication with pagination support"""
        return {
            "operationName": "PublicationPostsQuery",
            "variables": {
                "publicationId": publication_id,
                "first": limit,
                "after": cursor
            },
            "query": """query PublicationPostsQuery($publicationId: ID!, $first: Int, $after: String) {
                publication(id: $publicationId) {
                    id
                    name
                    posts(first: $first, after: $after) {
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                        edges {
                            node {
                                id
                                title
                                uniqueSlug
                                firstPublishedAt
                                __typename
                            }
                        }
                    }
                }
            }"""
        }
    
    def _parse_post(self, post_data: Dict[str, Any]) -> Post:
        """Parse post data from API response to domain entity"""
        author_data = post_data.get("creator", {})
        author = Author(
            id=author_data.get("id", ""),
            name=author_data.get("name", "Unknown"),
            username=author_data.get("username", "unknown")
        )
        
        # Parse timestamps
        published_timestamp = post_data.get("firstPublishedAt")
        latest_timestamp = post_data.get("latestPublishedAt")
        
        published_at = datetime.fromtimestamp(published_timestamp / 1000) if published_timestamp else datetime.now()
        latest_published_at = datetime.fromtimestamp(latest_timestamp / 1000) if latest_timestamp else None
        
        subtitle = post_data.get("extendedPreviewContent", {}).get("subtitle", "")
        
        return Post(
            id=PostId(post_data["id"]),
            title=post_data["title"],
            slug=post_data["uniqueSlug"],
            author=author,
            published_at=published_at,
            reading_time=post_data.get("readingTime", 0),
            subtitle=subtitle,
            latest_published_at=latest_published_at
        )
