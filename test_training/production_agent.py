# production_agent.py
import ollama
import json
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"

@dataclass
class AgentAction:
    tool: str
    input: str
    reasoning: str

@dataclass
class AgentObservation:
    result: str
    success: bool

class ProductionAgent:
    """Production-ready agent with proper state management"""
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
        self.conversation_history: List[Dict] = []
        self.state = AgentState.THINKING
        
    def register_tool(self, name: str, func: Callable, description: str):
        """Register a tool"""
        self.tools[name] = func
        self.tool_descriptions[name] = description
        
    def _get_system_prompt(self) -> str:
        tools = "\n".join([
            f"- {name}: {desc}"
            for name, desc in self.tool_descriptions.items()
        ])
        
        return f"""You are an AI agent that solves tasks step by step.

          Available tools:
          {tools}

          IMPORTANT: You must take ONE action at a time and wait for the result before taking the next action.

          Process:
          1. Think about what to do next
          2. Choose ONE tool and provide input
          3. Observe the result
          4. Repeat until task is complete

          When using a tool, respond with EXACTLY ONE JSON object and nothing else:
          {{
            "reasoning": "why you're using this tool",
            "tool": "tool_name",
            "input": "tool input"
          }}

          When task is complete, respond with EXACTLY ONE JSON object:
          {{
            "reasoning": "task completed because...",
            "complete": true,
            "answer": "final answer here"
          }}
          
          DO NOT return multiple JSON objects in one response. Take one action, wait for the result, then decide the next action."""
    
    def _parse_response(self, response: str) -> Dict:
        """Parse agent response - extract first complete JSON object"""
        # Try to find the first complete JSON object
        start = response.find('{')
        if start == -1:
            return {'error': 'No JSON found in response'}
        
        # Find the matching closing brace for the first opening brace
        brace_count = 0
        end = start
        
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if brace_count != 0:
            return {'error': 'No matching closing brace found'}
        
        json_str = response[start:end]
        
        try:
            parsed_json = json.loads(json_str)
            
            # If there are multiple JSON objects, warn about it
            remaining = response[end:].strip()
            if remaining and remaining.startswith('{'):
                if self.verbose:
                    print(f"\n⚠️  Warning: Multiple JSON objects detected. Using only the first one.")
                    print(f"Remaining content: {remaining[:100]}...")
            
            return parsed_json
        except json.JSONDecodeError as e:
            return {'error': f'JSON decode error: {str(e)}'}
    
    def run(self, task: str) -> str:
        """Execute a task"""
        self.conversation_history = []
        
        # Add initial task
        self.conversation_history.append({
            'role': 'user',
            'content': f"Task: {task}"
        })
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*60}")
            
            # Get agent response
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self._get_system_prompt()}
                ] + self.conversation_history
            )
            
            content = response['message']['content']
            
            if self.verbose:
                print(f"\nAgent Response:\n{content}")
            
            # Parse response
            parsed = self._parse_response(content)
            
            if 'error' in parsed:
                if self.verbose:
                    print(f"\nError: {parsed['error']}")
                continue
            
            # Check if complete
            if parsed.get('complete'):
                if self.verbose:
                    print(f"\n✓ Task Complete!")
                    print(f"Reasoning: {parsed.get('reasoning')}")
                return parsed.get('answer', content)
            
            # Execute tool
            tool_name = parsed.get('tool')
            tool_input = parsed.get('input')
            reasoning = parsed.get('reasoning', 'No reasoning provided')
            
            if self.verbose:
                print(f"\nReasoning: {reasoning}")
                print(f"Using tool: {tool_name}")
                print(f"Input: {tool_input}")
            
            if tool_name not in self.tools:
                observation = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    result = self.tools[tool_name](tool_input)
                    observation = f"Success: {result}"
                except Exception as e:
                    observation = f"Error: {str(e)}"
            
            if self.verbose:
                print(f"Observation: {observation}")
            
            # Add to conversation
            self.conversation_history.append({
                'role': 'assistant',
                'content': content
            })
            self.conversation_history.append({
                'role': 'user',
                'content': f"Observation: {observation}"
            })
        
        return "Max iterations reached without completing task"

# Example tools
def calculator(expression: str) -> str:
    """Safe calculator"""
    try:
        # Only allow basic math operations
        allowed = set('0123456789+-*/()%. ')
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def file_writer(content: str) -> str:
    """Write to a file"""
    try:
        # Parse "filename:content"
        filename, text = content.split(':', 1)
        with open(filename.strip(), 'w') as f:
            f.write(text.strip())
        return f"Successfully wrote to {filename}"
    except Exception as e:
        return f"File Write Error: {str(e)}"

def web_search_mock(query: str) -> str:
    """Mock web search (you can integrate real search here)"""
    return f"Search results for '{query}': [Mock results would go here]"

# Create and use agent
agent = ProductionAgent(model="llama3.2:3b", verbose=True)

agent.register_tool(
    "calculator",
    calculator,
    "Evaluate mathematical expressions. Input: a valid math expression"
)

agent.register_tool(
    "file_writer",
    file_writer,
    "Write content to a file. Input: 'filename.txt:content to write'"
)

agent.register_tool(
    "search",
    web_search_mock,
    "Search the web. Input: search query"
)

# Run a task
result = agent.run(
    "Calculate the area of a circle with radius 7.5, then write the result to circle_area.txt"
)

print(f"\n\nFinal Result: {result}")