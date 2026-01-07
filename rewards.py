import ast
import re

class RewardCalculator:
    def __init__(self):
        # Hyperparameters for shaping the reward signal
        self.R_COMPILE = 0.2      # Small reward for valid syntax
        self.R_PASS = 1.0         # Reward per passed test case
        self.R_FAIL = -0.5        # Penalty for failed test case
        self.R_ERROR = -1.0       # Penalty for runtime/syntax errors
        self.R_EFFICIENCY = -0.01 # Penalty per token (encourages concise code)

    def compute(self, code: str, output: str, exit_success: bool, num_tests: int = 1) -> float:
        """
        Calculates the reward scalar for a generated solution.
        
        Args:
            code: The Python source code generated.
            output: stdout/stderr captured from Docker.
            exit_success: True if the container exited with code 0.
            num_tests: Number of test cases run (used for normalization).
        """
        reward = 0.0

        # 1. Syntax Check (Static Analysis)
        # We reward the model just for writing valid Python, even if logic is wrong.
        if self._is_syntax_valid(code):
            reward += self.R_COMPILE
        else:
            # If syntax is invalid, heavily penalize and return immediately.
            return self.R_ERROR

        # 2. Functional Correctness
        if exit_success:
            # We assume 'Pass' logic creates stdout output like "Tests Passed!"
            # In a real scenario, you parse the specific number of passed tests from stdout.
            # For this simplified version, success exit code means all tests passed.
            reward += (self.R_PASS * num_tests)
        else:
            # Check for runtime errors vs logical failures
            if "Traceback" in output or "Error:" in output:
                reward += self.R_ERROR
            else:
                # Code ran but assertions failed
                reward += (self.R_FAIL * num_tests)

        # 3. Efficiency Penalty (Length Penalty)
        # Prevents the model from dumping huge comments or dead code to 'hack' the reward.
        # We tokenize by simple whitespace split for speed.
        num_tokens = len(code.split())
        reward += (num_tokens * self.R_EFFICIENCY)

        return reward

    def _is_syntax_valid(self, code: str) -> bool:
        """Checks if code parses into a valid AST."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

# --- Quick Test Block ---
if __name__ == "__main__":
    calc = RewardCalculator()

    # Scenario 1: Perfect Code
    code_good = "def add(a, b): return a + b"
    r_good = calc.compute(code_good, "Tests Passed", True, num_tests=5)
    print(f"Reward (Good): {r_good:.2f}")

    # Scenario 2: Syntax Error
    code_syntax = "def add(a, b) return a + b" # Missing colon
    r_syntax = calc.compute(code_syntax, "SyntaxError", False)
    print(f"Reward (Syntax Error): {r_syntax:.2f}")

    # Scenario 3: Runtime Error
    code_runtime = "def add(a, b): return a / 0"
    r_runtime = calc.compute(code_runtime, "ZeroDivisionError", False)
    print(f"Reward (Runtime Error): {r_runtime:.2f}")
