import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import redis
from datetime import datetime, timedelta

class LongTermMemory:
    """Long-term memory implementation - vector database storage"""
    
    def __init__(self, index_path: str = "memory_index", use_redis: bool = False):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the MiniLM model
        
        # FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        self.memory_store = {}  # Store full memory data
        self.index_path = index_path
        
        # Optional Redis cache
        self.redis_client = None
        if use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
            except:
                print("Redis connection failed, using in-memory storage")
                
    def store(self, text: str, metadata: Dict[str, Any], importance: float = 0.5) -> str:
        """Store memory in the vector database"""
        # Generate embedding vector
        embedding = self.model.encode([text])[0]
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        # Create memory ID
        memory_id = f"mem_{int(time.time() * 1000)}"
        
        # Store memory data
        memory_data = {
            "text": text,
            "metadata": {
                **metadata,
                "timestamp": time.time(),
                "importance": importance,
                "memory_id": memory_id
            },
            "embedding": embedding.tolist()
        }
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        self.memory_store[memory_id] = memory_data
        
        # Add to Redis cache (if available)
        if self.redis_client:
            self.redis_client.setex(
                f"memory:{memory_id}",
                timedelta(days=7),
                json.dumps(memory_data)
            )
            
        return memory_id
    
    def retrieve(self, query: str, k: int = 3, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve memory based on semantic similarity"""
        if self.index.ntotal == 0:
            return []
            
        # Generate query vector
        query_embedding = self.model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search for similar memories
        scores, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= threshold and idx != -1:
                memory_id = list(self.memory_store.keys())[idx]
                memory_data = self.memory_store[memory_id]
                results.append({
                    "text": memory_data["text"],
                    "metadata": memory_data["metadata"],
                    "similarity": float(score)
                })
                
        return results
    
    def retrieve_by_time_range(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Retrieve memories by time range"""
        results = []
        for memory_id, memory_data in self.memory_store.items():
            timestamp = memory_data["metadata"]["timestamp"]
            if start_time <= timestamp <= end_time:
                results.append({
                    "text": memory_data["text"],
                    "metadata": memory_data["metadata"]
                })
        return sorted(results, key=lambda x: x["metadata"]["timestamp"], reverse=True)
    
    def retrieve_by_importance(self, min_importance: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieve memories by importance"""
        results = []
        for memory_data in self.memory_store.values():
            if memory_data["metadata"]["importance"] >= min_importance:
                results.append({
                    "text": memory_data["text"],
                    "metadata": memory_data["metadata"]
                })
        return sorted(results, key=lambda x: x["metadata"]["importance"], reverse=True)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory"""
        if memory_id in self.memory_store:
            del self.memory_store[memory_id]
            # Note: FAISS index does not support direct deletion, needs to be rebuilt
            self._rebuild_index()
            return True
        return False
    
    def _rebuild_index(self):
        """Rebuild FAISS index"""
        self.index = faiss.IndexFlatIP(self.dimension)
        embeddings = []
        for memory_data in self.memory_store.values():
            embedding = np.array(memory_data["embedding"]).astype('float32')
            embeddings.append(embedding)
        
        if embeddings:
            embeddings = np.array(embeddings)
            self.index.add(embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        total_memories = len(self.memory_store)
        if total_memories == 0:
            return {"total_memories": 0}
            
        importances = [m["metadata"]["importance"] for m in self.memory_store.values()]
        timestamps = [m["metadata"]["timestamp"] for m in self.memory_store.values()]
        
        return {
            "total_memories": total_memories,
            "avg_importance": np.mean(importances),
            "max_importance": np.max(importances),
            "min_importance": np.min(importances),
            "oldest_memory": datetime.fromtimestamp(min(timestamps)).isoformat(),
            "newest_memory": datetime.fromtimestamp(max(timestamps)).isoformat()
        }
    
    def save_index(self):
        """Save index to file"""
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}.json", 'w') as f:
            json.dump(self.memory_store, f, default=str)
    
    def load_index(self):
        """Load index from file"""
        try:
            self.index = faiss.read_index(f"{self.index_path}.faiss")
            with open(f"{self.index_path}.json", 'r') as f:
                self.memory_store = json.load(f)
        except:
            print("Index file not found, creating new index")

class MemorySummarizer:
    """Memory summarization implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def generate_summary(self, messages: List[str], max_length: int = 200) -> str:
        """Generate conversation summary"""
        if not messages:
            return ""
            
        # Merge messages
        full_text = " ".join(messages)
        
        # Extract key sentences
        sentences = full_text.split('. ')
        if len(sentences) <= 3:
            return full_text[:max_length] + "..." if len(full_text) > max_length else full_text
            
        # Use embeddings to compute sentence importance
        sentence_embeddings = self.model.encode(sentences)
        text_embedding = self.model.encode([full_text])[0]
        
        # Compute similarity as importance score
        similarities = np.dot(sentence_embeddings, text_embedding) / (
            np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )
        
        # Select most important sentences
        top_indices = np.argsort(similarities)[-3:][::-1]
        summary_sentences = [sentences[i] for i in sorted(top_indices)]
        
        summary = ". ".join(summary_sentences)
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    
    def extract_key_points(self, messages: List[str]) -> Dict[str, List[str]]:
        """Extract key points"""
        full_text = " ".join(messages)
        
        # Simple keyword extraction
        key_points = {
            "facts": [],
            "agreements": [],
            "issues": []
        }
        
        # Extract facts with numbers
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', full_text)
        if numbers:
            key_points["facts"].extend([f"Contains number: {num}" for num in numbers[:3]])
        
        # Extract questions
        questions = re.findall(r'[^.!?]*\?[^.!?]*', full_text)
        if questions:
            key_points["issues"].extend(questions[:3])
            
        # Extract agreements (keywords like "agree", "okay", etc.)
        agreement_patterns = ["同意", "好的", "可以", "没问题"]
        for pattern in agreement_patterns:
            if pattern in full_text:
                key_points["agreements"].append(f"Agreement reached: {pattern}")
                
        return key_points
