"""
E-commerce toolset for the ReAct Agent (Lab 3).

Design principle (see INSTRUCTOR_GUIDE.md): an LLM only knows a tool through its
string `description`. Descriptions here are intentionally precise about the
expected argument names, types and value space so the model can call them
correctly. Each tool is a plain Python function; the `TOOLS` registry at the
bottom binds a name + description to the callable so the agent can dispatch
dynamically.
"""

from typing import Dict, Any, List

# --- Mock "databases" -------------------------------------------------------
# In a real system these would be API/DB calls. Keeping them in-memory makes the
# lab deterministic and easy to reason about when reading failure traces.

_PRICE_DB: Dict[str, float] = {
    "iphone": 999.0,
    "ipad": 599.0,
    "macbook": 1999.0,
    "airpods": 199.0,
}

_STOCK_DB: Dict[str, int] = {
    "iphone": 12,
    "ipad": 0,        # deliberately out of stock -> good failure-handling case
    "macbook": 4,
    "airpods": 250,
}

# coupon_code -> percentage discount (as a whole number, e.g. 10 == 10%)
_COUPON_DB: Dict[str, float] = {
    "WINNER": 10.0,
    "VIP20": 20.0,
    "BLACKFRIDAY": 35.0,
}

# destination -> flat shipping cost per kg (USD)
_SHIPPING_RATES: Dict[str, float] = {
    "hanoi": 2.0,
    "saigon": 2.5,
    "danang": 3.0,
}


def _normalize(name: str) -> str:
    return str(name).strip().strip("\"'").lower()


# --- Tools ------------------------------------------------------------------

def get_price(item_name: str) -> str:
    """Return the unit price (USD) of a single catalog item."""
    key = _normalize(item_name)
    if key not in _PRICE_DB:
        return f"Error: unknown item '{item_name}'. Available items: {', '.join(_PRICE_DB)}."
    return f"The unit price of {key} is ${_PRICE_DB[key]:.2f}."


def check_stock(item_name: str) -> str:
    """Return how many units of an item are currently in stock."""
    key = _normalize(item_name)
    if key not in _STOCK_DB:
        return f"Error: unknown item '{item_name}'. Available items: {', '.join(_STOCK_DB)}."
    qty = _STOCK_DB[key]
    if qty == 0:
        return f"{key} is OUT OF STOCK (0 units available)."
    return f"{key} has {qty} units in stock."


def get_discount(coupon_code: str) -> str:
    """Look up the discount percentage granted by a coupon code."""
    code = str(coupon_code).strip().strip("\"'").upper()
    if code not in _COUPON_DB:
        return f"Error: coupon '{coupon_code}' is invalid or expired."
    return f"Coupon {code} grants a {_COUPON_DB[code]:.0f}% discount."


def calc_shipping(weight: float, destination: str) -> str:
    """Estimate shipping cost (USD) given a weight in kg and a destination city."""
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return f"Error: weight must be a number, got '{weight}'."
    dest = _normalize(destination)
    if dest not in _SHIPPING_RATES:
        return f"Error: we do not ship to '{destination}'. Supported: {', '.join(_SHIPPING_RATES)}."
    cost = w * _SHIPPING_RATES[dest]
    return f"Shipping {w:.1f}kg to {dest} costs ${cost:.2f}."


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Supports + - * / ( ) and decimals only."""
    expr = str(expression).strip()
    allowed = set("0123456789.+-*/() ")
    if not expr or not set(expr) <= allowed:
        return f"Error: expression '{expression}' contains disallowed characters. Use only numbers and + - * / ( )."
    try:
        # Safe: the character whitelist above forbids names, attributes and calls.
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:  # noqa: BLE001 - surface the parse error to the agent
        return f"Error evaluating '{expression}': {e}"
    return f"{expr} = {result}"


# --- Registry ---------------------------------------------------------------
# Each entry: name (how the LLM addresses it), description (the ONLY thing the
# LLM sees), and func (the callable the agent dispatches to).

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_price",
        "description": "get_price(item_name: str) -> unit price in USD of one item. "
                       "Valid items: iphone, ipad, macbook, airpods.",
        "func": get_price,
    },
    {
        "name": "check_stock",
        "description": "check_stock(item_name: str) -> number of units available. "
                       "Valid items: iphone, ipad, macbook, airpods.",
        "func": check_stock,
    },
    {
        "name": "get_discount",
        "description": "get_discount(coupon_code: str) -> discount percentage for a coupon code "
                       "(e.g. WINNER, VIP20). Returns an error for invalid codes.",
        "func": get_discount,
    },
    {
        "name": "calc_shipping",
        "description": "calc_shipping(weight: float, destination: str) -> shipping cost in USD. "
                       "destination must be one of: hanoi, saigon, danang.",
        "func": calc_shipping,
    },
    {
        "name": "calculator",
        "description": "calculator(expression: str) -> evaluates an arithmetic expression. "
                       "Use this for ALL math (totals, applying discounts, tax). "
                       "Example: calculator(\"999 * 2 * 0.9\").",
        "func": calculator,
    },
]
