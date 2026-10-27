"""Optional third-party dependencies used by feature-gated APIs."""

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None


def require_pandas():
    if pd is None:
        raise RuntimeError("This feature requires pandas. Install it with: pip install pandas")


def require_matplotlib():
    if plt is None:
        raise RuntimeError("This feature requires matplotlib. Install it with: pip install matplotlib")


def require_prettytable():
    if PrettyTable is None:
        raise RuntimeError("This feature requires prettytable. Install it with: pip install prettytable")
