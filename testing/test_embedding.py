import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.text_embedding_small import get_embedding


def test_single_text():
    text = "Apple's Q2 earnings exceeded expectations."
    embedding = get_embedding(text)
    print("✅ Test 1 Passed")
    print(f"Vector length: {len(embedding)}")
    print(f"Preview: {embedding[:5]}")

def test_empty_text():
    text = "   "
    embedding = get_embedding(text)
    assert embedding == [], "Empty input should return empty embedding"
    print("✅ Test 2 Passed — Empty string handled correctly")

def test_long_text():
    text = "Revenue and margins were up this quarter due to improved supply chain..." * 10
    embedding = get_embedding(text)
    print("✅ Test 3 Passed — Long text handled")
    print(f"Vector length: {len(embedding)}")

if __name__ == "__main__":
    print("🔍 Testing text-embedding-3-small...")
    test_single_text()
    test_empty_text()
    test_long_text()
