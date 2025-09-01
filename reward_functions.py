"""
Comprehensive reward functions for GRPO training of Enhanced HRM
- Math verification (GSM8K, MATH)
- Code execution (APPS, HumanEval)
- Format rewards for reasoning structure
- Latent space reasoning rewards
"""

import re
import ast
import math
import torch
import sympy
import subprocess
import tempfile
import os
import signal
from typing import Dict, List, Tuple, Optional, Union
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")


class RewardFunction:
    """Base class for reward functions"""
    
    def __init__(self, weight: float = 1.0, name: str = "base"):
        self.weight = weight
        self.name = name
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        raise NotImplementedError
        
    def batch_compute(self, responses: List[str], references: Optional[List[str]] = None, **kwargs) -> List[float]:
        """Compute rewards for a batch of responses"""
        if references is None:
            references = [None] * len(responses)
        return [self(resp, ref, **kwargs) for resp, ref in zip(responses, references)]


class FormatReward(RewardFunction):
    """Reward for proper reasoning format with thinking tags"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight, "format")
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        reward = 0.0
        
        # Check for thinking tags
        has_think_tags = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
        has_solution_tags = bool(re.search(r'<SOLUTION>.*?</SOLUTION>', response, re.DOTALL))
        
        # Perfect format bonus
        if has_think_tags and has_solution_tags:
            reward += 3.0
        elif has_think_tags or has_solution_tags:
            reward += 1.0
            
        # Check for structured reasoning
        has_step_markers = bool(re.search(r'(step \d+|first|second|third|finally)', response.lower()))
        if has_step_markers:
            reward += 0.5
            
        # Penalize excessive thinking without solution
        think_content = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_content:
            think_length = sum(len(content.strip()) for content in think_content)
            if think_length > 500:  # Too verbose
                reward -= 0.5
            elif think_length < 20:  # Too brief
                reward -= 0.2
                
        # Solution should be concise
        solution_content = re.findall(r'<SOLUTION>(.*?)</SOLUTION>', response, re.DOTALL)
        if solution_content:
            solution_text = solution_content[0].strip()
            if len(solution_text) > 200:  # Too verbose answer
                reward -= 0.3
                
        return reward


class MathReward(RewardFunction):
    """Reward for mathematical correctness (GSM8K, MATH)"""
    
    def __init__(self, weight: float = 3.0):
        super().__init__(weight, "math")
        
    def extract_number(self, text: str) -> Optional[float]:
        """Extract the final numerical answer"""
        # Look for numbers in solution tags first
        solution_match = re.search(r'<SOLUTION>(.*?)</SOLUTION>', text, re.DOTALL)
        if solution_match:
            solution_text = solution_match.group(1)
        else:
            solution_text = text
            
        # Common number patterns
        patterns = [
            r'(?:answer is|equals?|=)\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)\s*(?:dollars?|cents?|\$)',
            r'\$?\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d+(?:\.\d+)?)\s*$',  # Number at end
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, solution_text.lower())
            if matches:
                try:
                    return float(matches[-1])  # Last match is usually the answer
                except ValueError:
                    continue
                    
        return None
        
    def normalize_answer(self, answer: Union[str, float]) -> Optional[float]:
        """Normalize answer for comparison"""
        if isinstance(answer, (int, float)):
            return float(answer)
            
        if isinstance(answer, str):
            # Remove common text
            answer = re.sub(r'[^\d\.\-\+]', '', answer)
            try:
                return float(answer)
            except ValueError:
                return None
                
        return None
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        if reference is None:
            return 0.0
            
        predicted = self.extract_number(response)
        expected = self.normalize_answer(reference)
        
        if predicted is None or expected is None:
            return 0.0
            
        # Exact match
        if abs(predicted - expected) < 1e-6:
            return 5.0
            
        # Close match (within 1%)
        if expected != 0 and abs(predicted - expected) / abs(expected) < 0.01:
            return 3.0
            
        # Reasonable ballpark (within 10%)
        if expected != 0 and abs(predicted - expected) / abs(expected) < 0.10:
            return 1.0
            
        return 0.0


class CodeReward(RewardFunction):
    """Reward for code correctness with execution"""
    
    def __init__(self, weight: float = 4.0, timeout: int = 5):
        super().__init__(weight, "code")
        self.timeout = timeout
        
    @contextmanager
    def timeout_context(self, seconds):
        """Context manager for timeout"""
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
            
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    def extract_code(self, response: str) -> Optional[str]:
        """Extract code from response"""
        # Look for code blocks
        code_patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'<SOLUTION>(.*?)</SOLUTION>',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                code = matches[-1].strip()
                if len(code) > 10:  # Minimum code length
                    return code
                    
        return None
        
    def test_code(self, code: str, test_cases: List[Dict]) -> Tuple[int, int]:
        """Test code against test cases"""
        if not code:
            return 0, len(test_cases)
            
        passed = 0
        total = len(test_cases)
        
        for test_case in test_cases:
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    # Write code and test
                    test_code = f"""
{code}

# Test case
try:
    {test_case.get('test', 'assert False')}
    print("PASSED")
except Exception as e:
    print(f"FAILED: {{e}}")
"""
                    f.write(test_code)
                    f.flush()
                    
                    # Execute with timeout
                    with self.timeout_context(self.timeout):
                        result = subprocess.run([
                            'python', f.name
                        ], capture_output=True, text=True, timeout=self.timeout)
                        
                        if "PASSED" in result.stdout:
                            passed += 1
                            
            except (TimeoutError, subprocess.TimeoutExpired):
                pass  # Timeout = fail
            except Exception:
                pass  # Any error = fail
            finally:
                try:
                    os.unlink(f.name)
                except:
                    pass
                    
        return passed, total
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        code = self.extract_code(response)
        if not code:
            return 0.0
            
        # Get test cases from kwargs
        test_cases = kwargs.get('test_cases', [])
        if not test_cases:
            # Basic syntax check
            try:
                ast.parse(code)
                return 1.0  # Valid Python syntax
            except SyntaxError:
                return 0.0
                
        # Execute tests
        passed, total = self.test_code(code, test_cases)
        
        if total == 0:
            return 1.0 if code else 0.0
            
        # Reward based on pass rate
        pass_rate = passed / total
        if pass_rate >= 1.0:
            return 5.0
        elif pass_rate >= 0.8:
            return 3.0
        elif pass_rate >= 0.5:
            return 1.5
        elif pass_rate > 0:
            return 0.5
        else:
            return 0.0


class LatentReasoningReward(RewardFunction):
    """Reward for effective use of HRM's latent reasoning cycles"""
    
    def __init__(self, weight: float = 0.5):
        super().__init__(weight, "latent_reasoning")
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        # Get HRM-specific metrics
        cycles_used = kwargs.get('cycles_used', 1)
        consistency_loss = kwargs.get('consistency_loss', 0.0)
        confidence = kwargs.get('confidence', 0.5)
        
        reward = 0.0
        
        # Efficiency reward - penalize excessive cycles for simple tasks
        response_length = len(response.split())
        optimal_cycles = min(6, max(1, response_length // 20))  # Heuristic
        
        if cycles_used <= optimal_cycles:
            reward += 1.0
        elif cycles_used > optimal_cycles * 1.5:
            reward -= 0.5  # Penalty for inefficiency
            
        # Consistency reward - stable reasoning across cycles
        if isinstance(consistency_loss, (int, float)):
            if consistency_loss < 0.1:
                reward += 0.5
            elif consistency_loss > 0.5:
                reward -= 0.3
                
        # Confidence calibration
        if isinstance(confidence, (int, float)):
            if 0.6 <= confidence <= 0.9:  # Well-calibrated confidence
                reward += 0.3
            elif confidence < 0.3:  # Too uncertain
                reward -= 0.2
                
        return reward


class BrevityReward(RewardFunction):
    """Reward for concise solutions after thinking"""
    
    def __init__(self, weight: float = 0.3):
        super().__init__(weight, "brevity")
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        # Extract solution part
        solution_match = re.search(r'<SOLUTION>(.*?)</SOLUTION>', response, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
            solution_length = len(solution.split())
            
            # Reward concise but complete solutions
            if 5 <= solution_length <= 30:
                return 1.0
            elif solution_length > 50:
                return -0.5  # Penalty for verbosity
                
        return 0.0


class RewardCombiner:
    """Combines multiple reward functions with weights"""
    
    def __init__(self, reward_functions: List[RewardFunction]):
        self.reward_functions = reward_functions
        
    def compute_rewards(
        self, 
        responses: List[str], 
        references: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[List[float], Dict[str, List[float]]]:
        """Compute combined rewards and individual component rewards"""
        
        if references is None:
            references = [None] * len(responses)
            
        # Compute individual rewards
        component_rewards = {}
        for reward_fn in self.reward_functions:
            rewards = reward_fn.batch_compute(responses, references, **kwargs)
            component_rewards[reward_fn.name] = rewards
            
        # Combine rewards
        total_rewards = []
        for i in range(len(responses)):
            total_reward = sum(
                reward_fn.weight * component_rewards[reward_fn.name][i] 
                for reward_fn in self.reward_functions
            )
            total_rewards.append(total_reward)
            
        return total_rewards, component_rewards


def create_math_reward_combiner() -> RewardCombiner:
    """Create reward combiner for math problems"""
    return RewardCombiner([
        FormatReward(weight=1.0),
        MathReward(weight=5.0),
        LatentReasoningReward(weight=0.5),
        BrevityReward(weight=0.3)
    ])


def create_code_reward_combiner() -> RewardCombiner:
    """Create reward combiner for coding problems"""
    return RewardCombiner([
        FormatReward(weight=1.0),
        CodeReward(weight=5.0),
        LatentReasoningReward(weight=0.5),
        BrevityReward(weight=0.3)
    ])


if __name__ == "__main__":
    # Test reward functions
    print("🧪 Testing reward functions...")
    
    # Test math reward
    math_reward = MathReward()
    response_good = "<think>Let me solve: 2+2</think><SOLUTION>4</SOLUTION>"
    response_bad = "<think>Hmm...</think><SOLUTION>5</SOLUTION>"
    
    print(f"Math reward (correct): {math_reward(response_good, '4')}")
    print(f"Math reward (wrong): {math_reward(response_bad, '4')}")
    
    # Test format reward
    format_reward = FormatReward()
    print(f"Format reward (good): {format_reward(response_good)}")
    print(f"Format reward (bad): {format_reward('just 4')}")
    
    # Test code reward
    code_reward = CodeReward()
    code_response = """<think>Need to write a function</think>
<SOLUTION>
def add(a, b):
    return a + b
</SOLUTION>"""
    
    test_cases = [
        {'test': 'assert add(2, 3) == 5'},
        {'test': 'assert add(0, 0) == 0'}
    ]
    
    print(f"Code reward: {code_reward(code_response, test_cases=test_cases)}")
    
    # Test combiner
    combiner = create_math_reward_combiner()
    responses = [response_good, response_bad]
    references = ['4', '4']
    
    total_rewards, component_rewards = combiner.compute_rewards(responses, references)
    print(f"Combined rewards: {total_rewards}")
    print(f"Component rewards: {component_rewards}")
    
    print("✅ Reward functions working!")