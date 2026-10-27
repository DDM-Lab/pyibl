"""Shared internal constants for PyIBL runtime modules."""

LEGEND_LIMIT = 10

AGGREGATE_COLUMNS = tuple((
    "iteration,time,choice,utility,option,blended_value,"
    "retrieval_probability,activation,base_level_activation,"
    "activation_noise"
).split(","))

PLOT_COLORS = "blue,green,red,black,magenta,orange,cyan".split(",")
PLOT_LINE_STYLES = (
    "-",
    "--",
    ":",
    "-.",
    (0, (3, 6)),
    (5, (10, 3)),
    (0, (3, 2, 1, 2)),
    (0, (3, 3, 2, 3)),
)
