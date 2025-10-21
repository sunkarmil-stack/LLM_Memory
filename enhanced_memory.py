# memory_system.py

import time
import numpy as np
from typing import List, Dict, Any, Optional
import json
import threading
from collections import defaultdict
import re


class DynamicRetriever:
    """Dynamic Memory Retrieval System"""

    def __init__(self, long_term_memory, short_term_memory):
        self.lt_memory = long_term_memory
        self.st_memory = short_term_memory
        self.retrieval_cache = {}
        self.cache_ttl = 300  # 5-minute cache

    def hierarchical_retrieval(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Hierarchical retrieval: Semantic → Temporal → Importance"""
        cache_key = f"{query}_{max_results}"

        # Check cache
        if cache_key in self.retrieval_cache:
            cached_result, timestamp = self.retrieval_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result

        # Get all memories
        all_memories = []

        # Retrieve from long-term memory
        lt_results = self.lt_memory.retrieve(query, k=max_results * 2)
        all_memories.extend(lt_results)

        # Retrieve from short-term memory
        st_context = self.st_memory.get_context()
        if st_context:
            all_memories.append({
                "text": st_context,
                "metadata": {
                    "source": "short_term",
                    "timestamp": time.time(),
                    "importance": 0.8,
                    "type": "recent_conversation"
                }
            })

        # Apply hierarchical filtering
        results = self._apply_hierarchical_filtering(all_memories, query, max_results)

        # Cache results
        self.retrieval_cache[cache_key] = (results, time.time())

        return results

    def _apply_hierarchical_filtering(self, memories: List[Dict[str, Any]],
                                      query: str, max_results: int) -> List[Dict[str, Any]]:
        """Apply hierarchical filtering"""
        if not memories:
            return []

        # Level 1: Semantic relevance
        query_embedding = self.lt_memory.model.encode([query])[0]
        semantic_scores = []

        for memory in memories:
            text = memory.get('text', '')
            if text:
                text_embedding = self.lt_memory.model.encode([text])[0]
                similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
                )
                semantic_scores.append((memory, similarity))

        # Sort by semantic score and take top 60%
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        top_semantic = semantic_scores[:max(1, int(len(semantic_scores) * 0.6))]

        # Level 2: Time weighting
        time_weighted = []
        current_time = time.time()

        for memory, semantic_score in top_semantic:
            metadata = memory.get('metadata', {})
            timestamp = metadata.get('timestamp', current_time)
            days_old = (current_time - timestamp) / (24 * 3600)

            # Time decay function: linear decay within 30 days
            time_weight = max(0.1, 1.0 - (days_old / 30))
            time_weighted.append((memory, semantic_score * time_weight))

        # Level 3: Importance filtering
        final_results = []
        for memory, combined_score in time_weighted:
            metadata = memory.get('metadata', {})
            importance = metadata.get('importance', 0.5)

            # Final score: semantic * time * importance
            final_score = combined_score * importance

            if final_score > 0.3:  # Threshold filtering
                memory['retrieval_score'] = final_score
                final_results.append(memory)

        # Sort by final score and limit results
        final_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return final_results[:max_results]

    def contextual_retrieval(self, query: str, context: Dict[str, Any],
                             max_results: int = 3) -> List[Dict[str, Any]]:
        """Context-aware intelligent retrieval"""
        # Extract context information
        user_profile = context.get('user_profile', {})
        recent_topics = context.get('recent_topics', [])
        conversation_history = context.get('conversation_history', [])

        # Build enhanced query
        enhanced_query = self._build_enhanced_query(query, context)

        # Perform retrieval
        base_results = self.hierarchical_retrieval(enhanced_query, max_results * 2)

        # Apply context weighting
        weighted_results = []
        for result in base_results:
            context_score = self._calculate_context_score(result, context)
            final_score = result.get('retrieval_score', 0.5) * (1 + context_score)

            result['context_score'] = context_score
            result['final_score'] = final_score
            weighted_results.append(result)

        # Re-sort results
        weighted_results.sort(key=lambda x: x['final_score'], reverse=True)
        return weighted_results[:max_results]

    def _build_enhanced_query(self, query: str, context: Dict[str, Any]) -> str:
        """Build enhanced query"""
        enhanced_parts = [query]

        # Add user profile keywords
        user_profile = context.get('user_profile', {})
        if user_profile:
            profile_keywords = f"User profile: {json.dumps(user_profile, ensure_ascii=False)}"
            enhanced_parts.append(profile_keywords)

        # Add recent topics
        recent_topics = context.get('recent_topics', [])
        if recent_topics:
            topics_str = f"Related topics: {', '.join(recent_topics[-3:])}"
            enhanced_parts.append(topics_str)

        return " ".join(enhanced_parts)

    def _calculate_context_score(self, memory: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate context relevance score"""
        score = 0.0

        # User profile matching
        user_profile = context.get('user_profile', {})
        memory_metadata = memory.get('metadata', {})

        if user_profile and 'user_id' in memory_metadata:
            if memory_metadata.get('user_id') == user_profile.get('user_id'):
                score += 0.5

        # Topic matching
        recent_topics = context.get('recent_topics', [])
        memory_text = memory.get('text', '').lower()

        for topic in recent_topics:
            if topic.lower() in memory_text:
                score += 0.3

        return min(score, 1.0)


class MemoryReflection:
    """Memory Reflection Mechanism"""

    def __init__(self, long_term_memory):
        self.lt_memory = long_term_memory
        self.reflection_schedule = {}
        self.reflection_thread = None
        self.running = False

    def start_periodic_reflection(self, interval_hours: int = 24):
        """Start periodic memory reflection"""
        self.running = True
        self.reflection_thread = threading.Thread(
            target=self._reflection_worker,
            args=(interval_hours,),
            daemon=True
        )
        self.reflection_thread.start()

    def stop_periodic_reflection(self):
        """Stop periodic memory reflection"""
        self.running = False

    def _reflection_worker(self, interval_hours: int):
        """Reflection worker thread"""
        while self.running:
            try:
                self.perform_reflection()
                time.sleep(interval_hours * 3600)
            except Exception as e:
                print(f"Memory reflection error: {e}")

    def perform_reflection(self, days_threshold: int = 7):
        """Perform memory reflection"""
        print("Starting memory reflection check...")

        # Get all memories
        all_memories = list(self.lt_memory.memory_store.values())

        # Filter old memories
        current_time = time.time()
        cutoff_time = current_time - (days_threshold * 24 * 3600)

        old_memories = [
            m for m in all_memories
            if m['metadata']['timestamp'] < cutoff_time
        ]

        # Re-evaluate importance
        updated_count = 0
        deleted_count = 0

        for memory in old_memories:
            # Recalculate importance
            text = memory['text']
            new_importance = self._recalculate_importance(text, memory['metadata'])

            memory_id = memory['metadata']['memory_id']

            if new_importance < 0.3:
                # Delete low-importance memories
                self.lt_memory.delete(memory_id)
                deleted_count += 1
            else:
                # Update importance
                memory['metadata']['importance'] = new_importance
                memory['metadata']['last_reflected'] = current_time
                updated_count += 1

        print(f"Memory reflection completed: Updated {updated_count}, Deleted {deleted_count}")

    def _recalculate_importance(self, text: str, metadata: Dict[str, Any]) -> float:
        """Recalculate memory importance"""
        # Adjust based on access frequency
        access_count = metadata.get('access_count', 0)
        base_importance = metadata.get('importance', 0.5)

        # Frequency boost
        frequency_boost = min(access_count * 0.05, 0.3)

        # Time decay
        days_old = (time.time() - metadata['timestamp']) / (24 * 3600)
        time_decay = max(0.1, 1.0 - (days_old / 30))

        new_importance = (base_importance + frequency_boost) * time_decay

        return min(new_importance, 1.0)

    def generate_memory_insights(self) -> Dict[str, Any]:
        """Generate memory insights report"""
        stats = self.lt_memory.get_stats()

        if stats['total_memories'] == 0:
            return {"message": "No memory data available"}

        # Get memories from the last 7 days
        week_ago = time.time() - (7 * 24 * 3600)
        recent_memories = self.lt_memory.retrieve_by_time_range(week_ago, time.time())

        # Analyze memory topics
        topics = self._analyze_memory_topics(recent_memories)

        # Generate insights
        insights = {
            "total_memories": stats['total_memories'],
            "recent_memories": len(recent_memories),
            "avg_importance": stats['avg_importance'],
            "key_topics": topics[:5],
            "memory_growth_rate": len(recent_memories) / 7,  # Average per day
            "recommendations": self._generate_recommendations(stats, recent_memories)
        }

        return insights

    def _analyze_memory_topics(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Analyze memory topics"""
        if not memories:
            return []

        # Simple keyword extraction
        all_text = " ".join([m['text'] for m in memories])
        words = re.findall(r'\b\w+\b', all_text.lower())

        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3 and word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all'}:
                word_freq[word] += 1

        # Return high-frequency words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]

    def _generate_recommendations(self, stats: Dict[str, Any],
                                  recent_memories: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if stats['avg_importance'] < 0.5:
            recommendations.append("Consider increasing the memory importance threshold")

        if len(recent_memories) > 50:
            recommendations.append("Memory growth is too rapid; consider increasing cleanup frequency")

        if stats['total_memories'] > 1000:
            recommendations.append("Memory volume is large; consider implementing tiered storage")

        return recommendations


class MultimodalMemory:
    """Multimodal Memory Support (Framework Implementation)"""

    def __init__(self):
        self.text_encoder = None  # Text encoder
        self.image_encoder = None  # Image encoder
        self.audio_encoder = None  # Audio encoder

    def store_multimodal(self, text: str = None, image_path: str = None,
                         audio_path: str = None, metadata: Dict[str, Any] = None):
        """Store multimodal memory"""
        memory_data = {
            'metadata': metadata or {},
            'timestamp': time.time(),
            'modalities': {}
        }

        if text:
            memory_data['modalities']['text'] = text

        if image_path:
            memory_data['modalities']['image'] = image_path

        if audio_path:
            memory_data['modalities']['audio'] = audio_path

        return memory_data

    def cross_modal_retrieval(self, query: str, modality: str = 'text') -> List[Dict[str, Any]]:
        """Cross-modal retrieval"""
        # Framework implementation; actual usage requires integration with respective encoders
        return []