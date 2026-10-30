import pytest
from src.retrieval.hybrid_search import reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    """Verify that RRF combines two result lists accurately."""
    list1 = [
        {"id": "doc1", "text": "Poshan 2.0 guidelines", "metadata": {}},
        {"id": "doc2", "text": "PM KISAN installment", "metadata": {}},
    ]
    list2 = [
        {"id": "doc2", "text": "PM KISAN installment", "metadata": {}},
        {"id": "doc3", "text": "PMAY Urban housing", "metadata": {}},
    ]

    fused = reciprocal_rank_fusion([list1, list2], k=60, top_n=3)
    assert len(fused) == 3
    # doc2 appears in both lists, so its fused score should be highest
    assert fused[0]["id"] == "doc2"
    assert "rrf_score" in fused[0]
