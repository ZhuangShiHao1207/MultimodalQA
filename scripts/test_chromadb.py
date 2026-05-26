"""Quick test: ChromaDB VectorStore basic operations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.indexing.embedder import VectorStore
from src.ingestion.models import DocumentElement, ElementType

# Test 1: Create store + add + search
print("Test 1: Create, add, search...")
store = VectorStore(collection_name='test_migration', persist_dir='./test_chroma')
elem = DocumentElement(type=ElementType.TEXT, text_content='hello world test', page_number=1)
vec = np.random.randn(1, 1024).astype(np.float32)
store.add_elements([elem], vec)
results = store.search(vec[0], top_k=1)
print(f"  Search: {len(results)} hits, score={results[0][1]:.4f}")
print(f"  Store size: {store.size}")

# Test 2: Persistence - create new client, data should still be there
print("\nTest 2: Persistence check...")
store2 = VectorStore(collection_name='test_migration', persist_dir='./test_chroma')
print(f"  Reloaded store size: {store2.size}")
assert store2.size == 1, "Persistence failed!"
results2 = store2.search(vec[0], top_k=1)
print(f"  Search after reload: {len(results2)} hits")

# Cleanup
store2.delete_collection()
import shutil
shutil.rmtree('./test_chroma', ignore_errors=True)
print("\nAll tests passed!")
