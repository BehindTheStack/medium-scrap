"""
Infrastructure Adapters - External System Integration
Following Adapter Pattern and Dependency Inversion Principle
"""

import time
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from ...domain.entities.publication import (
    Post, PostId, PublicationConfig, PublicationType, Author, PublicationId
)
from ...domain.repositories.base import PostRepository, PublicationRepository, SessionRepository
from ..http_transport import HTTPTransport, HttpxTransport


class MediumApiAdapter(PostRepository):
    """
    Adapter for Medium GraphQL API
    Implements Repository interface for external API access
    """
    
    def __init__(self, transport: Optional[HTTPTransport] = None):
        self._base_headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "apollographql-client-name": "lite",
            "apollographql-client-version": "main-20251110-155029-e7a0736aac",
            "medium-frontend-app": "lite/main-20251110-155029-e7a0736aac",
            "sec-ch-ua": '"Chromium";v="131", "Google Chrome";v="131", "Not_A Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin"
        }
        # Allow injection of a transport for testability; default to HttpxTransport
        self._transport = transport or HttpxTransport()

    def _safe_http_post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]):
        """Delegate POST to the injected transport for testable HTTP calls."""
        # Add graphql-operation header if operationName is present in payload
        if "operationName" in payload:
            headers = headers.copy()
            headers["graphql-operation"] = payload["operationName"]
        return self._transport.post(url, headers=headers, json=payload)
    
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
        
        # For usernames (starts with @), skip publication query and use GraphQL directly
        if config.id.value.startswith('@'):
            try:
                # Strategy 1: GraphQL User Query (for usernames)
                discovered_ids = self._discover_via_graphql(config, limit)
                if discovered_ids:
                    return discovered_ids
            except Exception:
                pass
            
            try:
                # Strategy 2: HTML Scraping fallback
                discovered_ids = self._discover_via_html_scraping(config, limit)
                if discovered_ids:
                    return discovered_ids
            except Exception:
                pass
        else:
            # For publications, try publication query first
            try:
                # Strategy 1: Publication All Posts Query (NEW)
                discovered_ids = self._discover_via_publication_all(config, limit)
                if discovered_ids:
                    return discovered_ids
            except Exception:
                pass
            
            try:
                # Strategy 2: GraphQL Publication Query
                discovered_ids = self._discover_via_graphql(config, limit)
                if discovered_ids:
                    return discovered_ids
            except Exception:
                pass
            
            try:
                # Strategy 3: HTML Scraping
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
            response = self._safe_http_post(config.graphql_url, headers, query)

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
        max_to_collect = limit if limit is not None else 999999  # Use a large number instead of infinity
        
        # Rate limiting - wait before making request
        # time.sleep(1.0)  # 1 second delay to avoid 429 errors
        
        while collected < max_to_collect:
            remaining = max_to_collect - collected if limit is not None else page_size
            current_page_size = min(page_size, remaining if limit is not None else page_size)
            
            query = self._build_publication_query(config.id.value, current_page_size, cursor)
            
            try:
                response = self._safe_http_post(config.graphql_url, headers, query)

                if response.status_code == 429:
                    # Rate limited - wait and retry once
                    time.sleep(2.0)
                    response = self._safe_http_post(config.graphql_url, headers, query)
                    if response.status_code != 200:
                        break
                elif response.status_code != 200:
                    break

                data = response.json()
                if data.get("errors"):
                    break

                # Handle different response structures for users vs publications
                if config.id.value.startswith('@'):
                    # User profile response structure
                    if (not data.get("data") or
                        not data["data"].get("userResult") or
                        not data["data"]["userResult"].get("homepagePostsConnection")):
                        break

                    posts_connection = data["data"]["userResult"]["homepagePostsConnection"]
                    posts_list = posts_connection.get("posts", [])

                    if not posts_list:
                        break

                    # Collect post IDs from this page
                    page_ids = []
                    for post in posts_list:
                        try:
                            page_ids.append(PostId(post["id"]))
                        except Exception:
                            # Skip invalid post IDs
                            continue
                    
                    all_post_ids.extend(page_ids)
                    collected += len(page_ids)

                    # Check pagination info
                    paging_info = posts_connection.get("pagingInfo", {})
                    next_page = paging_info.get("next")

                    if not next_page or not next_page.get("from"):
                        break

                    cursor = next_page.get("from")

                else:
                    # Publication response structure (original)
                    if (not data.get("data") or
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
        max_to_collect = limit if limit is not None else 999999
        
        while len(all_ids) < max_to_collect:
            if config.is_custom_domain:
                # For custom domains, try different page formats
                if page == 0:
                    url = f"https://{config.domain}"
                else:
                    url = f"https://{config.domain}/?page={page}"
            else:
                # For Medium-hosted publications (users and publications)
                if page == 0:
                    url = f"https://medium.com/{config.id.value}"
                else:
                    url = f"https://medium.com/{config.id.value}?page={page}"
            
            try:
                response = self._transport.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }, follow_redirects=True)
                
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
    
    def _discover_via_publication_all(self, config: PublicationConfig, limit: Optional[int]) -> List[PostId]:
        """Discover post IDs using the /all route and PublicationContentDataQuery"""
        all_post_ids = []
        cursor = None
        collected = 0
        max_to_collect = limit if limit is not None else 999999
        
        # Build the publication reference based on configuration
        if config.is_custom_domain:
            publication_ref = {
                "slug": None,
                "domain": config.domain
            }
        else:
            # For Medium-hosted publications, use slug
            publication_ref = {
                "slug": config.id.value,
                "domain": None
            }
        
        headers = self._get_headers_for_config(config)
        
        # Start with initial query (no cursor)
        page = 0
        max_pages = 200 if limit is None else 20  # More pages for --all
        while collected < max_to_collect and page < max_pages:
            try:
                # Build the query
                remaining_to_collect = max_to_collect - collected if max_to_collect != 999999 else 25
                variables = {
                    "ref": publication_ref,
                    "first": min(25, remaining_to_collect),
                    "after": cursor if cursor else "",  # Empty string for first page
                    "orderBy": {"publishedAt": "DESC"},
                    "filter": {"published": True}
                }
                
                query_payload = {
                    "operationName": "PublicationContentDataQuery",
                    "variables": variables,
                    "query": """query PublicationContentDataQuery($ref: PublicationRef!, $first: Int!, $after: String!, $orderBy: PublicationPostsOrderBy, $filter: PublicationPostsFilter) {
  publication: publicationByRef(ref: $ref) {
    __typename
    id
    publicationPostsConnection(
      first: $first
      after: $after
      orderBy: $orderBy
      filter: $filter
    ) {
      __typename
      edges {
        listedAt
        node {
          id
          title
                    uniqueSlug
                    canonicalUrl
                    tags
                    virtuals { totalClapCount }
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
      pageInfo {
        endCursor
        hasNextPage
        __typename
      }
    }
  }
}"""
                }

                response = self._safe_http_post(config.graphql_url, headers, query_payload)

                if response.status_code != 200:
                    break
                
                data = response.json()
                
                if ("errors" in data or 
                    not data.get("data") or 
                    not data["data"].get("publication") or 
                    not data["data"]["publication"].get("publicationPostsConnection")):
                    break
                
                posts_connection = data["data"]["publication"]["publicationPostsConnection"]
                edges = posts_connection.get("edges", [])
                
                if not edges:
                    break
                
                # Collect post IDs from this page
                valid_post_ids = []
                invalid_ids = []
                for edge in edges:
                    if edge.get("node"):
                        post_id = edge["node"]["id"]
                        try:
                            valid_post_ids.append(PostId(post_id))
                        except ValueError as e:
                            invalid_ids.append(post_id)
                            # Continue processing even with invalid IDs
                            continue
                
                all_post_ids.extend(valid_post_ids)
                collected += len(valid_post_ids)
                
                # Check pagination
                page_info = posts_connection.get("pageInfo", {})
                if not page_info.get("hasNextPage"):
                    break
                    
                cursor = page_info.get("endCursor")
                if not cursor:
                    break
                
                page += 1
                time.sleep(0.3)  # Rate limiting
                
            except Exception:
                break
        
        return all_post_ids
    
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
        # Check if it's a user profile (starts with @)
        if publication_id.startswith('@'):
            username = publication_id[1:]  # Remove @ for the query
            return {
                "operationName": "UserProfileQuery",
                "variables": {
                    "homepagePostsFrom": cursor,
                    "includeDistributedResponses": True,
                    "id": None,
                    "username": username,
                    "homepagePostsLimit": limit
                },
                "query": """query UserProfileQuery($id: ID, $username: ID, $homepagePostsLimit: PaginationLimit, $homepagePostsFrom: String = null, $includeDistributedResponses: Boolean = true) {
                    userResult(id: $id, username: $username) {
                        __typename
                        ... on User {
                            id
                            name
                            homepagePostsConnection(
                                paging: {limit: $homepagePostsLimit, from: $homepagePostsFrom}
                                includeDistributedResponses: $includeDistributedResponses
                            ) {
                                posts {
                                    id
                                    title
                                    uniqueSlug
                                    canonicalUrl
                                    tags
                                    virtuals { totalClapCount }
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
                                pagingInfo {
                                    next {
                                        from
                                        limit
                                        __typename
                                    }
                                    __typename
                                }
                                __typename
                            }
                            __typename
                        }
                    }
                }"""
            }
        else:
            # Original publication query
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
                                    canonicalUrl
                                    tags
                                    virtuals { totalClapCount }
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

        published_at = datetime.fromtimestamp(published_timestamp / 1000, tz=timezone.utc) if published_timestamp else datetime.now(timezone.utc)
        latest_published_at = datetime.fromtimestamp(latest_timestamp / 1000, tz=timezone.utc) if latest_timestamp else None
        
        subtitle = post_data.get("extendedPreviewContent", {}).get("subtitle", "")
        
        return Post(
            id=PostId(post_data["id"]),
            title=post_data["title"],
            slug=post_data["uniqueSlug"],
            author=author,
            published_at=published_at,
            reading_time=post_data.get("readingTime", 0),
            subtitle=subtitle,
            latest_published_at=latest_published_at,
            # Populate optional metadata when available
            url=post_data.get('canonicalUrl') or f"https://medium.com/p/{post_data.get('id')}",
            tags=post_data.get('tags') or [],
            claps=(post_data.get('virtuals') or {}).get('totalClapCount')
        )

    def fetch_post_html(self, post: Post, config: PublicationConfig) -> Optional[str]:
        """Fetch the full HTML content for a given post URL.

        Uses a best-effort approach: for medium-hosted posts we use the short 'p' URL; for custom domains we
        try the domain + slug path.
        
        Includes retry logic for rate limiting (429).
        """
        import time
        
        max_retries = 5
        base_delay = 3  # Start with 3 seconds
        
        for attempt in range(max_retries):
            try:
                if config.is_custom_domain:
                    url = f"https://{config.domain}/{post.slug}"
                else:
                    # Use the short 'p' URL which reliably redirects to the canonical post on medium
                    url = f"https://medium.com/p/{post.id.value}"

                response = self._transport.get(url, headers={"User-Agent": self._base_headers.get("user-agent", "")}, follow_redirects=True)

                if response.status_code == 200:
                    return response.text
                elif response.status_code in (429, 403):
                    # Rate limited or blocked - retry with exponential backoff (silent to not interfere with Rich Progress)
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 3,6,12...
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final failure - silent
                        return None
                else:
                    # Other HTTP errors - silent
                    return None
                    
            except Exception as e:
                # Silent to not interfere with progress bar
                return None
        
        return None
