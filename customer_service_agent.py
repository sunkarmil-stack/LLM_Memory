# customer_service_agent.py

import time
import json
from typing import Dict, Any, List, Optional

from short_term_memory import ShortTermMemory
from long_term_memory import LongTermMemory
from memory_router import MemoryRouter, ImportanceScorer
from enhanced_memory import DynamicRetriever, MemoryReflection


class CustomerServiceAgent:
    """Customer Service Agent with Memory – Full Conversational Agent"""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.st_memory = ShortTermMemory(window_size=20)
        self.lt_memory = LongTermMemory()
        self.memory_router = MemoryRouter(importance_threshold=0.6)
        self.importance_scorer = ImportanceScorer()
        self.dynamic_retriever = DynamicRetriever(self.lt_memory, self.st_memory)
        self.memory_reflection = MemoryReflection(self.lt_memory)

        # User profiles
        self.user_profiles = {}

        # Conversation statistics
        self.conversation_stats = {
            'total_messages': 0,
            'long_term_stored': 0,
            'memory_retrievals': 0
        }

    def respond(self, user_id: str, message: str) -> Dict[str, Any]:
        """Generate a response"""
        self.conversation_stats['total_messages'] += 1

        # Retrieve user profile
        profile = self.user_profiles.get(user_id, {})

        # Build context
        context = self._build_context(user_id, message, profile)

        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(message, user_id)

        # Generate response
        response = self._generate_response(message, context, relevant_memories)

        # Update memory
        self._update_memory(user_id, message, response)

        return {
            'response': response,
            'context_used': len(relevant_memories),
            'memory_stats': self.conversation_stats,
            'user_profile': profile
        }

    def _build_context(self, user_id: str, message: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Build full context"""
        return {
            'user_profile': profile,
            'recent_topics': self._extract_recent_topics(),
            'conversation_history': self.st_memory.get_recent_messages(5),
            'current_message': message,
            'user_id': user_id
        }

    def _extract_recent_topics(self) -> List[str]:
        """Extract recent topics"""
        recent_text = self.st_memory.get_context()
        if not recent_text:
            return []

        # Simple keyword extraction
        words = recent_text.lower().split()
        topics = [word for word in words if len(word) > 4 and word.isalpha()]
        return list(set(topics))[:5]

    def _retrieve_relevant_memories(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories"""
        self.conversation_stats['memory_retrievals'] += 1

        # Build context
        context = {
            'user_profile': self.user_profiles.get(user_id, {}),
            'recent_topics': self._extract_recent_topics()
        }

        # Use dynamic retrieval
        memories = self.dynamic_retriever.contextual_retrieval(
            query, context, max_results=3
        )

        return memories

    def _generate_response(self, message: str, context: Dict[str, Any],
                          memories: List[Dict[str, Any]]) -> str:
        """Generate response (simulated)"""
        # Build prompt
        memory_context = "\n".join([
            f"- {m['text']}" for m in memories[-2:]
        ]) if memories else "No relevant historical memory"

        prompt = f"""Generate a helpful reply based on the following information:

User profile: {json.dumps(context['user_profile'], ensure_ascii=False)}
Relevant memories: {memory_context}
Conversation history: {context['conversation_history'][-2:] if context['conversation_history'] else 'None'}
User message: {message}

Please provide a helpful response:"""

        # Simplified handling; in practice, this would call an LLM
        response = f"I understand your request. Based on our previous conversation, {message}. Let me assist you."

        if memories:
            response += f" I recall some relevant information: {memories[0]['text'][:100]}..."

        return response

    def _update_memory(self, user_id: str, message: str, response: str):
        """Update memory system"""
        full_interaction = f"User: {message}\nAgent: {response}"

        # Calculate importance
        context = {
            'user_profile': self.user_profiles.get(user_id, {}),
            'recent_topics': self._extract_recent_topics()
        }

        importance = self.importance_scorer.calculate_importance(full_interaction, context)

        # Store in long-term memory
        if importance >= 0.6:
            metadata = {
                'user_id': user_id,
                'timestamp': time.time(),
                'importance': importance,
                'type': 'customer_service_interaction',
                'message_length': len(message) + len(response)
            }

            self.lt_memory.store(full_interaction, metadata)
            self.conversation_stats['long_term_stored'] += 1

        # Update short-term memory
        self.st_memory.add(f"User: {message}")
        self.st_memory.add(f"Agent: {response}")

        # Update user profile
        self._update_user_profile(user_id, message, response)

    def _update_user_profile(self, user_id: str, message: str, response: str):
        """Update user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'first_interaction': time.time(),
                'total_interactions': 0,
                'common_topics': [],
                'preferences': {}
            }

        profile = self.user_profiles[user_id]
        profile['total_interactions'] += 1
        profile['last_interaction'] = time.time()

        # Extract topics
        topics = self._extract_topics_from_text(message + " " + response)
        profile['common_topics'].extend(topics)
        profile['common_topics'] = list(set(profile['common_topics']))[-10:]

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Simple keyword extraction
        keywords = ['order', 'product', 'service', 'issue', 'help', 'suggestion', 'complaint', 'inquiry']
        topics = [kw for kw in keywords if kw in text]
        return topics

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get user insights"""
        profile = self.user_profiles.get(user_id, {})
        if not profile:
            return {"message": "User not found"}

        # Retrieve user-related memories
        user_memories = self.lt_memory.retrieve(f"user:{user_id}", k=10)

        insights = {
            'user_profile': profile,
            'total_memories': len(user_memories),
            'interaction_frequency': profile.get('total_interactions', 0),
            'common_topics': profile.get('common_topics', []),
            'last_interaction': profile.get('last_interaction', 0),
            'memory_summary': self._summarize_user_memories(user_memories)
        }

        return insights

    def _summarize_user_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Summarize user memories"""
        if not memories:
            return "No memories available"

        # Extract key information
        key_points = []
        for memory in memories[-3:]:
            text = memory.get('text', '')
            if len(text) > 50:
                key_points.append(text[:100] + "...")

        return " | ".join(key_points)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'conversation_stats': self.conversation_stats,
            'total_users': len(self.user_profiles),
            'memory_stats': self.lt_memory.get_stats(),
            'reflection_insights': self.memory_reflection.generate_memory_insights()
        }

    def cleanup_old_memories(self, days_threshold: int = 30):
        """Clean up old memories"""
        self.memory_reflection.perform_reflection(days_threshold)

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data"""
        profile = self.user_profiles.get(user_id, {})
        memories = self.lt_memory.retrieve(f"user:{user_id}", k=100)

        return {
            'user_profile': profile,
            'memories': memories,
            'export_timestamp': time.time(),
            'data_version': '1.0'
        }


class ConversationSimulator:
    """Conversation Simulator – For Testing"""

    def __init__(self, agent: CustomerServiceAgent):
        self.agent = agent

    def simulate_conversation(self, user_id: str, conversation: List[str]):
        """Simulate a conversation"""
        responses = []

        for message in conversation:
            response = self.agent.respond(user_id, message)
            responses.append(response)
            print(f"User: {message}")
            print(f"Agent: {response['response']}")
            print("-" * 50)

        return responses

    def run_demo(self):
        """Run demonstration"""
        print("=== Customer Service Memory System Demo ===\n")

        # Simulate user conversations
        conversations = {
            "user_001": [
                "I'd like to check the status of my order",
                "My order number is 12345. When will it arrive?",
                "I'm very satisfied with your product and would like to buy another one",
                "The return issue I asked about earlier has been resolved. Thank you."
            ],
            "user_002": [
                "I can't log into my account",
                "I haven't received the password reset email",
                "It's working now. Thanks for your help"
            ]
        }

        for user_id, messages in conversations.items():
            print(f"\n--- Conversation for User {user_id} ---")
            self.simulate_conversation(user_id, messages)

        # Display statistics
        stats = self.agent.get_system_stats()
        print("\n=== System Statistics ===")
        print(f"Total messages: {stats['conversation_stats']['total_messages']}")
        print(f"Long-term memories stored: {stats['conversation_stats']['long_term_stored']}")
        print(f"Memory retrievals: {stats['conversation_stats']['memory_retrievals']}")
        print(f"Total users: {len(self.agent.user_profiles)}")

        # Display user insights
        for user_id in conversations.keys():
            insights = self.agent.get_user_insights(user_id)
            print(f"\n--- Insights for User {user_id} ---")
            print(json.dumps(insights, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Create customer service agent
    agent = CustomerServiceAgent()

    # Create simulator
    simulator = ConversationSimulator(agent)

    # Run demo
    simulator.run_demo()