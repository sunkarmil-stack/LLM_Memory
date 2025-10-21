# app.py

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import time
from customer_service_agent import CustomerServiceAgent

app = Flask(__name__)
CORS(app)

# Create global customer service agent instance
agent = CustomerServiceAgent()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Long- and Short-Term Memory System Demo</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.bootcdn.net/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">
                <i class="fas fa-brain text-blue-600"></i>
                LLM Long- and Short-Term Memory System
            </h1>
            <p class="text-gray-600">Complete implementation demo based on blog post</p>
        </div>

        <!-- System Status -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-white rounded-lg shadow p-4">
                <h3 class="text-lg font-semibold mb-2">
                    <i class="fas fa-chart-bar text-green-600"></i>
                    System Stats
                </h3>
                <div id="system-stats" class="text-sm text-gray-600">
                    Loading...
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow p-4">
                <h3 class="text-lg font-semibold mb-2">
                    <i class="fas fa-users text-blue-600"></i>
                    User Management
                </h3>
                <select id="user-select" class="w-full p-2 border rounded">
                    <option value="user_001">User 001</option>
                    <option value="user_002">User 002</option>
                    <option value="user_003">User 003</option>
                </select>
            </div>
        </div>

        <!-- Chat Area -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Chat Panel -->
            <div class="lg:col-span-2">
                <div class="bg-white rounded-lg shadow">
                    <div class="p-4 border-b">
                        <h2 class="text-xl font-semibold">
                            <i class="fas fa-comments text-blue-600"></i>
                            Conversation Demo
                        </h2>
                    </div>
                    
                    <div class="p-4">
                        <div id="chat-messages" class="h-96 overflow-y-auto mb-4 border rounded p-3 bg-gray-50">
                            <div class="text-center text-gray-500 py-8">
                                <i class="fas fa-robot text-4xl mb-2"></i>
                                <p>Start chatting to experience the memory system</p>
                            </div>
                        </div>
                        
                        <div class="flex gap-2">
                            <input type="text" id="message-input" 
                                   class="flex-1 p-2 border rounded" 
                                   placeholder="Type a message..."
                                   onkeypress="handleKeyPress(event)">
                            <button onclick="sendMessage()" 
                                    class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        
                        <div class="mt-4 flex gap-2">
                            <button onclick="loadDemo()" 
                                    class="px-3 py-1 bg-green-600 text-white rounded text-sm">
                                Load Demo
                            </button>
                            <button onclick="clearMemory()" 
                                    class="px-3 py-1 bg-red-600 text-white rounded text-sm">
                                Clear Memory
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Memory Panel -->
            <div class="lg:col-span-1">
                <div class="bg-white rounded-lg shadow">
                    <div class="p-4 border-b">
                        <h2 class="text-xl font-semibold">
                            <i class="fas fa-database text-purple-600"></i>
                            Memory Status
                        </h2>
                    </div>
                    
                    <div class="p-4">
                        <!-- Short-Term Memory -->
                        <div class="mb-4">
                            <h3 class="font-semibold text-sm text-gray-700 mb-2">Short-Term Memory</h3>
                            <div id="short-term-memory" class="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                                None
                            </div>
                        </div>
                        
                        <!-- Long-Term Memory -->
                        <div class="mb-4">
                            <h3 class="font-semibold text-sm text-gray-700 mb-2">Long-Term Memory</h3>
                            <div id="long-term-memory" class="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                                None
                            </div>
                        </div>
                        
                        <!-- User Insights -->
                        <div>
                            <h3 class="font-semibold text-sm text-gray-700 mb-2">User Insights</h3>
                            <div id="user-insights" class="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                                Select a user to view insights
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Technical Notes -->
        <div class="mt-8 bg-white rounded-lg shadow p-6">
            <h2 class="text-2xl font-semibold mb-4">
                <i class="fas fa-info-circle text-blue-600"></i>
                Technical Implementation Notes
            </h2>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                    <h3 class="font-semibold text-green-600 mb-2">Short-Term Memory Features</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li>Context window management</li>
                        <li>Circular buffer optimization</li>
                        <li>Attention mechanism enhancement</li>
                        <li>Dynamic window resizing</li>
                    </ul>
                </div>
                
                <div>
                    <h3 class="font-semibold text-purple-600 mb-2">Long-Term Memory Features</h3>
                    <ul class="list-disc list-inside text-gray-600 space-y-1">
                        <li>Vector database storage</li>
                        <li>Memory summarization</li>
                        <li>Importance scoring algorithm</li>
                        <li>Memory reflection mechanism</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentUser = 'user_001';

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadSystemStats();
            loadUserInsights();
        });

        // Handle user selection change
        document.getElementById('user-select').addEventListener('change', function(e) {
            currentUser = e.target.value;
            loadUserInsights();
        });

        // Send message
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessageToChat('user', message);
            input.value = '';
            
            fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: currentUser, message: message})
            })
            .then(response => response.json())
            .then(data => {
                addMessageToChat('agent', data.response);
                updateMemoryDisplay();
                updateSystemStats();
            });
        }

        // Add message to chat
        function addMessageToChat(sender, message) {
            const chatDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `mb-2 ${sender === 'user' ? 'text-right' : 'text-left'}`;
            
            const bubble = document.createElement('div');
            bubble.className = `inline-block px-3 py-2 rounded-lg max-w-xs ${
                sender === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-800'
            }`;
            bubble.textContent = message;
            
            messageDiv.appendChild(bubble);
            chatDiv.appendChild(messageDiv);
            chatDiv.scrollTop = chatDiv.scrollHeight;
        }

        // Load system stats
        function loadSystemStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const statsDiv = document.getElementById('system-stats');
                    statsDiv.innerHTML = `
                        <div>Total Messages: ${data.conversation_stats.total_messages}</div>
                        <div>Long-Term Memories: ${data.conversation_stats.long_term_stored}</div>
                        <div>Retrievals: ${data.conversation_stats.memory_retrievals}</div>
                        <div>Total Users: ${data.total_users}</div>
                    `;
                });
        }

        // Update system stats
        function updateSystemStats() {
            loadSystemStats();
        }

        // Update memory display
        function updateMemoryDisplay() {
            // Update short-term memory
            fetch('/api/short-term')
                .then(response => response.json())
                .then(data => {
                    const div = document.getElementById('short-term-memory');
                    div.innerHTML = data.messages || 'None';
                });

            // Update long-term memory
            fetch(`/api/long-term?user_id=${currentUser}`)
                .then(response => response.json())
                .then(data => {
                    const div = document.getElementById('long-term-memory');
                    div.innerHTML = data.memories || 'None';
                });
        }

        // Load user insights
        function loadUserInsights() {
            fetch(`/api/user-insights/${currentUser}`)
                .then(response => response.json())
                .then(data => {
                    const div = document.getElementById('user-insights');
                    if (data.message) {
                        div.innerHTML = data.message;
                    } else {
                        div.innerHTML = `
                            <div>Interactions: ${data.interaction_frequency}</div>
                            <div>Common Topics: ${data.common_topics.join(', ')}</div>
                            <div>Memory Count: ${data.total_memories}</div>
                        `;
                    }
                });
            
            updateMemoryDisplay();
        }

        // Load demo conversation
        function loadDemo() {
            const demoMessages = [
                "I'd like to check my order status",
                "My order number is 12345. When will it arrive?",
                "I'm very satisfied with your product",
                "The issue I asked about earlier has been resolved"
            ];
            
            let index = 0;
            function sendNext() {
                if (index < demoMessages.length) {
                    document.getElementById('message-input').value = demoMessages[index];
                    sendMessage();
                    index++;
                    setTimeout(sendNext, 2000);
                }
            }
            sendNext();
        }

        // Clear memory
        function clearMemory() {
            fetch('/api/clear-memory', {method: 'POST'})
                .then(() => {
                    location.reload();
                });
        }

        // Handle Enter key press
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    message = data.get('message', '')
    
    response = agent.respond(user_id, message)
    return jsonify(response)

@app.route('/api/stats')
def stats():
    return jsonify(agent.get_system_stats())

@app.route('/api/short-term')
def short_term():
    messages = agent.st_memory.get_recent_messages(5)
    return jsonify({'messages': '<br>'.join(m for m in messages)})

@app.route('/api/long-term')
def long_term():
    user_id = request.args.get('user_id', 'default_user')
    memories = agent.lt_memory.retrieve(f"user:{user_id}", k=3)
    
    memory_text = []
    for m in memories:
        text = m.get('text', '')[:100] + '...' if len(m.get('text', '')) > 100 else m.get('text', '')
        memory_text.append(text)
    
    return jsonify({'memories': '<br>'.join(memory_text) if memory_text else 'None'})

@app.route('/api/user-insights/<user_id>')
def user_insights(user_id):
    insights = agent.get_user_insights(user_id)
    return jsonify(insights)

@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    # Reset agent
    global agent
    agent = CustomerServiceAgent()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)