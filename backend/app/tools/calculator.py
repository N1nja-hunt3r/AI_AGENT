"""Safe calculator module: AST-based expression evaluation, arithmetic,
scientific functions, error handling, and health checks."""

from __future__ import annotations

import ast
import math
import operator
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class CalculatorError(Exception):
    """Base error for calculator operations."""


class InvalidExpressionError(CalculatorError):
    """Raised when an expression is malformed, empty, or too long."""


class UnsafeExpressionError(CalculatorError):
    """Raised when an expression contains disallowed syntax or names."""


class DivisionByZeroError(CalculatorError):
    """Raised when a division or modulo operation divides by zero."""


class DomainMathError(CalculatorError):
    """Raised when a math operation is outside its valid domain."""


class OverflowCalculatorError(CalculatorError):
    """Raised when a result is too large to represent as a finite float."""


# ----------------------------------------------------------------------
# Safe function / constant / operator whitelists
# ----------------------------------------------------------------------

def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1.0 / 3.0), x)


def _factorial(x: float) -> float:
    if x < 0 or not float(x).is_integer():
        raise ValueError("factorial() requires a non-negative integer.")
    return float(math.factorial(int(x)))


def _gcd(a: float, b: float) -> float:
    if not float(a).is_integer() or not float(b).is_integer():
        raise ValueError("gcd() requires integer arguments.")
    return float(math.gcd(int(a), int(b)))


SAFE_FUNCTIONS: Dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "cbrt": _cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "factorial": _factorial,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
    "gcd": _gcd,
}

SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

_ALLOWED_BINOPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# ----------------------------------------------------------------------
# Safe expression evaluator
# ----------------------------------------------------------------------

class SafeEvaluator:
    """Evaluates arithmetic/scientific expressions via a whitelisted AST walk."""

    MAX_EXPRESSION_LENGTH = 200

    def evaluate(self, expression: str) -> float:
        if not isinstance(expression, str) or not expression.strip():
            raise InvalidExpressionError("Expression must be a non-empty string.")
        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise InvalidExpressionError(
                f"Expression exceeds maximum length of {self.MAX_EXPRESSION_LENGTH} characters."
            )

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise InvalidExpressionError(f"Invalid syntax: {exc}") from exc

        try:
            result = self._eval_node(tree.body)
        except ZeroDivisionError as exc:
            raise DivisionByZeroError("Division or modulo by zero.") from exc
        except OverflowError as exc:
            raise OverflowCalculatorError("Result is too large to represent.") from exc
        except ValueError as exc:
            raise DomainMathError(f"Math domain error: {exc}") from exc
        except RecursionError as exc:
            raise InvalidExpressionError("Expression is too deeply nested.") from exc

        if isinstance(result, complex):
            raise DomainMathError("Result is a complex number; not supported.")

        result = float(result)
        if math.isnan(result):
            raise DomainMathError("Result is NaN.")
        if math.isinf(result):
            raise OverflowCalculatorError("Result is infinite.")
        return result

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsafeExpressionError("Only numeric literals are allowed.")
            return float(node.value)

        if isinstance(node, ast.BinOp):
            op_type: type[ast.operator] | type[ast.unaryop] = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise UnsafeExpressionError(f"Operator '{op_type.__name__}' is not allowed.")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return _ALLOWED_BINOPS[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise UnsafeExpressionError(f"Unary operator '{op_type.__name__}' is not allowed.")
            return _ALLOWED_UNARYOPS[op_type](self._eval_node(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("Only direct function calls are allowed.")
            func_name = node.func.id
            if func_name not in SAFE_FUNCTIONS:
                raise UnsafeExpressionError(f"Function '{func_name}' is not allowed.")
            if node.keywords:
                raise UnsafeExpressionError("Keyword arguments are not allowed.")
            args = [self._eval_node(arg) for arg in node.args]
            try:
                return float(SAFE_FUNCTIONS[func_name](*args))
            except TypeError as exc:
                raise InvalidExpressionError(
                    f"Invalid arguments for '{func_name}': {exc}"
                ) from exc

        if isinstance(node, ast.Name):
            if node.id not in SAFE_CONSTANTS:
                raise UnsafeExpressionError(f"Name '{node.id}' is not allowed.")
            return SAFE_CONSTANTS[node.id]

        raise UnsafeExpressionError(f"Expression element '{type(node).__name__}' is not allowed.")


# ----------------------------------------------------------------------
# Health check model
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CalculatorHealth:
    healthy: bool
    checks: Dict[str, bool]
    latency_ms: float
    timestamp: float
    detail: Optional[str] = None


# ----------------------------------------------------------------------
# Calculator
# ----------------------------------------------------------------------

class Calculator:
    """Typed calculator providing basic arithmetic, scientific functions
    (via safe expression evaluation), error handling, and health checks.
    """

    def __init__(self) -> None:
        self._evaluator = SafeEvaluator()

    # ------------------------------------------------------------------
    # Safe expression evaluation
    # ------------------------------------------------------------------

    def evaluate(self, expression: str) -> float:
        return self._evaluator.evaluate(expression)

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise DivisionByZeroError("Cannot divide by zero.")
        return a / b

    def modulo(self, a: float, b: float) -> float:
        if b == 0:
            raise DivisionByZeroError("Cannot compute modulo with a zero divisor.")
        return math.fmod(a, b)

    def power(self, base: float, exponent: float) -> float:
        try:
            result = math.pow(base, exponent)
        except ValueError as exc:
            raise DomainMathError(f"Invalid power operation: {exc}") from exc
        except OverflowError as exc:
            raise OverflowCalculatorError(f"Power result overflowed: {exc}") from exc
        if math.isinf(result):
            raise OverflowCalculatorError("Power result is infinite.")
        return result

    # ------------------------------------------------------------------
    # Scientific functions
    # ------------------------------------------------------------------

    def sqrt(self, value: float) -> float:
        if value < 0:
            raise DomainMathError("Cannot take the square root of a negative number.")
        return math.sqrt(value)

    def log(self, value: float, base: Optional[float] = None) -> float:
        try:
            return math.log(value) if base is None else math.log(value, base)
        except ValueError as exc:
            raise DomainMathError(f"Invalid logarithm operation: {exc}") from exc

    def sin(self, value_radians: float) -> float:
        return math.sin(value_radians)

    def cos(self, value_radians: float) -> float:
        return math.cos(value_radians)

    def tan(self, value_radians: float) -> float:
        return math.tan(value_radians)

    def factorial(self, value: float) -> float:
        try:
            return _factorial(value)
        except ValueError as exc:
            raise DomainMathError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self) -> CalculatorHealth:
        start = time.monotonic()
        checks: Dict[str, bool] = {}
        detail: Optional[str] = None

        try:
            checks["addition"] = self.add(2.0, 3.0) == 5.0
            checks["subtraction"] = self.subtract(5.0, 3.0) == 2.0
            checks["multiplication"] = self.multiply(4.0, 5.0) == 20.0
            checks["division"] = math.isclose(self.divide(10.0, 4.0), 2.5)
            checks["power"] = math.isclose(self.power(2.0, 10.0), 1024.0)
            checks["sqrt"] = math.isclose(self.sqrt(81.0), 9.0)
            checks["evaluation"] = math.isclose(
                self.evaluate("sin(0) + cos(0) * 2 + sqrt(16)"), 6.0, abs_tol=1e-9
            )
            checks["error_handling"] = self._check_raises(
                lambda: self.divide(1.0, 0.0), DivisionByZeroError
            )
        except CalculatorError as exc:
            detail = str(exc)

        healthy = bool(checks) and all(checks.values()) and detail is None
        latency_ms = (time.monotonic() - start) * 1000.0
        return CalculatorHealth(
            healthy=healthy,
            checks=checks,
            latency_ms=latency_ms,
            timestamp=time.time(),
            detail=detail,
        )

    @staticmethod
    def _check_raises(fn: Callable[[], float], expected: type) -> bool:
        try:
            fn()
        except expected:  # type: ignore[misc]
            return True
        except Exception:
            return False
        return False
