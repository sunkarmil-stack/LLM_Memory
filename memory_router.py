import time
from typing import List, Dict, Any
import numpy as np

class ShortTermMemory:
    """Short-term memory implementation - manages the dialogue context window"""
    
    def __init__(self, window_size: int = 10):
        self.memory = []
        self.window_size = window_size
        self.token_counts = []
        self.importance_scores = []
        
    def add(self, message: str, importance: float = 0.5) -> None:
        """Add a message to short-term memory using a circular buffer strategy"""
        if len(self.memory) >= self.window_size:
            self.memory.pop(0)
            self.token_counts.pop(0)
            self.importance_scores.pop(0)
            
        self.memory.append(message)
        self.token_counts.append(len(message.split()))
        self.importance_scores.append(importance)
        
    def get_context(self, max_tokens: int = 1000) -> str:
        """Retrieve context with token limit support"""
        total_tokens = 0
        context_messages = []
        
        # Start from the most recent messages, sorted by importance
        indexed_messages = list(zip(self.memory, self.importance_scores))
        indexed_messages.sort(key=lambda x: x[1], reverse=True)
        
        for message, _ in indexed_messages:
            message_tokens = len(message.split())
            if total_tokens + message_tokens <= max_tokens:
                context_messages.append(message)
                total_tokens += message_tokens
            else:
                break
                
        return "\n".join(reversed(context_messages))
    
    def get_recent_context(self, count: int = 5) -> str:
        """Retrieve the most recent few messages"""
        recent = self.memory[-count:] if len(self.memory) >= count else self.memory
        return "\n".join(recent)
    
    def clear(self) -> None:
        """Clear short-term memory"""
        self.memory.clear()
        self.token_counts.clear()
        self.importance_scores.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "message_count": len(self.memory),
            "total_tokens": sum(self.token_counts),
            "avg_importance": np.mean(self.importance_scores) if self.importance_scores else 0,
            "window_utilization": len(self.memory) / self.window_size
        }

class PositionalEncoding:
    """Positional encoding implementation for preserving sequence information"""
    
    @staticmethod
    def encode(seq_len: int, d_model: int) -> np.ndarray:
        """Generate positional encoding matrix"""
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe

class AttentionEnhancedMemory:
    """Short-term memory enhanced with attention mechanism"""
    
    def __init__(self, embed_dim: int = 512):
        self.embed_dim = embed_dim
        self.memory_embeddings = []
        self.memory_texts = []
        
    def add_message(self, text: str, embedding: np.ndarray):
        """Add a message with its embedding"""
        self.memory_texts.append(text)
        self.memory_embeddings.append(embedding)
        
    def get_context_with_attention(self, query_embedding: np.ndarray, top_k: int = 5) -> str:
        """Retrieve context based on attention weights"""
        if not self.memory_embeddings:
            return ""
            
        # Compute attention weights
        embeddings = np.array(self.memory_embeddings)
        attention_weights = np.dot(embeddings, query_embedding) / np.sqrt(self.embed_dim)
        
        # Get top-k relevant messages
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        context_messages = [self.memory_texts[i] for i in top_indices]
        return "\n".join(context_messages)
