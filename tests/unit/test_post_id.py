"""
Unit tests for PostId entity and validation logic
"""

import pytest
from src.domain.entities.publication import PostId


class TestPostIdValidation:
    """Test PostId validation (discovered during Netflix debugging)"""
    
    def test_valid_post_id_creation(self):
        """Test valid PostId creation"""
        # Test 12-character post IDs (the valid format)
        valid_id = "6dcc91058d8d"
        post_id = PostId(valid_id)
        assert post_id.value == valid_id
        assert len(post_id.value) == 12
    
    def test_invalid_post_id_rejection(self):
        """Test that invalid PostIds are rejected"""
        # Test short IDs (should raise ValueError)
        with pytest.raises(ValueError):
            PostId("short")
        
        # Test empty IDs
        with pytest.raises(ValueError):
            PostId("")
    
    def test_post_id_list_filtering(self):
        """Test that we can filter valid vs invalid IDs"""
        test_ids = ["6dcc91058d8d", "short", "33073e260a38", "", "47ef7af7ca2e"]
        valid_ids = []
        
        for test_id in test_ids:
            try:
                valid_ids.append(PostId(test_id))
            except ValueError:
                continue
        
        # Should get 3 valid IDs
        assert len(valid_ids) == 3
        assert all(len(pid.value) == 12 for pid in valid_ids)
