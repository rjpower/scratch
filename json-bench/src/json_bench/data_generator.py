"""Generate test documents for benchmarking."""

import random
import string
from datetime import datetime, timedelta


def _random_text(length: int) -> str:
    """Generate random text of approximately the given length."""
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
        "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
        "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat",
        "nulla", "pariatur", "excepteur", "sint", "occaecat", "cupidatat",
        "non", "proident", "sunt", "culpa", "qui", "officia", "deserunt",
        "mollit", "anim", "id", "est", "laborum", "data", "processing",
        "machine", "learning", "artificial", "intelligence", "neural", "network",
        "deep", "training", "model", "optimization", "performance", "benchmark",
    ]

    result = []
    current_length = 0
    while current_length < length:
        word = random.choice(words)
        result.append(word)
        current_length += len(word) + 1  # +1 for space

    return " ".join(result)


def generate_document(doc_id: int | None = None) -> dict:
    """
    Generate a single test document.

    The document contains:
    - ~1000 character body text
    - 10 metadata fields with various types
    - Nested structures (authors, tags, metrics)

    Args:
        doc_id: Optional document ID. If None, a random ID is generated.

    Returns:
        A dictionary representing the document.
    """
    if doc_id is None:
        doc_id = random.randint(1, 1_000_000)

    base_date = datetime(2020, 1, 1)
    created_at = base_date + timedelta(days=random.randint(0, 1825))

    return {
        # Primary fields
        "id": doc_id,
        "title": _random_text(50),
        "body": _random_text(1000),

        # 10 metadata fields
        "author": {
            "name": f"Author {random.randint(1, 100)}",
            "email": f"author{random.randint(1, 100)}@example.com",
            "affiliation": random.choice(["University A", "Lab B", "Company C", "Institute D"]),
        },
        "created_at": created_at.isoformat(),
        "updated_at": (created_at + timedelta(days=random.randint(0, 365))).isoformat(),
        "version": random.randint(1, 10),
        "status": random.choice(["draft", "published", "archived", "review"]),
        "priority": random.randint(1, 5),
        "category": random.choice(["research", "tutorial", "news", "analysis", "opinion"]),
        "language": random.choice(["en", "es", "fr", "de", "zh", "ja"]),
        "word_count": random.randint(500, 2000),
        "read_time_minutes": random.randint(3, 15),

        # Nested arrays
        "tags": [_random_text(10) for _ in range(random.randint(3, 8))],
        "related_ids": [random.randint(1, 1_000_000) for _ in range(random.randint(0, 5))],

        # Nested metrics
        "metrics": {
            "views": random.randint(0, 100000),
            "likes": random.randint(0, 5000),
            "shares": random.randint(0, 1000),
            "comments": random.randint(0, 500),
            "score": round(random.uniform(0, 5), 2),
        },
    }


def generate_documents(count: int) -> list[dict]:
    """
    Generate multiple test documents.

    Args:
        count: Number of documents to generate.

    Returns:
        A list of document dictionaries.
    """
    return [generate_document(i) for i in range(count)]
