# Copyright 2014-2026 Carnegie Mellon University

"""Plotting utilities for PyIBL aggregate data."""

import math
import numbers
from collections import Counter, defaultdict
from ._constants import AGGREGATE_COLUMNS, LEGEND_LIMIT, PLOT_COLORS, PLOT_LINE_STYLES
from ._optional_deps import pd, plt, require_matplotlib, require_pandas


def df_plot(df, kind, title=None, xlabel=None, ylabel=None,
            include=None, exclude=None, min=None, max=None, earliest=None, latest=None,
            legend=None, limits=None, filename=None, show=None):
    """This is similar to the :meth:`Agent.plot` method.
    The first argument should be a Pandas DataFrame, congruent to one such as might be
    produced by :attr:`Agent.aggrregate_details`. The plots produced are the same as would
    have been produced by an :class:`Agent` with those aggregate details, with the rest of
    the arguments being the same as for :meth:`Agent.plot`.

    This function can sometimes be useful in specialized circumstances, such as
    combining the results from multiple simulations using isomorphic Agents.

    A :exc:`ValueError` is raised if *df* is not a DataFrame, or its columns are
    such as might be in one produced by :attr:`Agent.aggrregate_details`. A variety
    of other errors may be raised if values in *df* are not of the types or ranges
    that might be expected in an :class:`Agent`'s results.
    """
    require_pandas()
    require_matplotlib()
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"First argument to df_plot must be a DataFrame, not {df}")
    for c in df.columns:
        if not (c in AGGREGATE_COLUMNS + ("mismatch",) or c.endswith(".similarity")):
            raise ValueError(f"DataFrame column {c} cannot be plotted")
    for c in AGGREGATE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"The DataFrame does not contain columns {c} and cannot be plotted")
    if min and not isinstance(min, numbers.Real):
        raise ValueError(f"The min value, {min}, is neither a Real number nor None")
    if max and not isinstance(max, numbers.Real):
        raise ValueError(f"The max value, {max}, is neither a Real number nor None")
    if show is None:
        show = filename is None
    plot_kind = {"choice":      ChoicePlot("choice", "Fraction making choice", True),
                 "bv":          OptionPlot("blended_value", "Mean blended value"),
                 "probability": RetrievalProbabilyPlot("retrieval_probability",
                                                       "Mean probability of retrieval", True),
                 "activation":  InstancePlot("activation", "Mean total activation"),
                 "noise":       InstancePlot("activation_noise", "Mean activation noise"),
                 "baselevel":   InstancePlot("base_level_activation", "Mean base level activation"),
                 "mismatch":    InstancePlot("mismatch", "Mean total mismatch penalty")
                 }.get(kind)
    if not plot_kind:
        if isinstance(kind, str) and kind.endswith(".similarity"):
            if kind in df:
                plot_kind = InstancePlot(kind, f"Mean similarities of {kind[0:-11]}",  True)
            else:
                raise ValueError(f"The {kind[0:-11]} attribute is either absent or not partially matched")
        else:
            raise ValueError(f"Unknown plot kind {kind}")
    if kind == "mismatch" and "mismatch" not in df:
        raise ValueError("Can't generate a mismatch plot when no attributes were partially matched")
    if xlabel is None:
        xlabel = "Time"
    if ylabel is None:
        ylabel = plot_kind._description
    data = plot_kind.get_data(df, include, exclude, min, max, earliest, latest)
    colors_and_styles = _plot_ordering(df, data)
    plt.clf()
    mint = None
    maxt = None
    for k in (colors_and_styles or data.keys()):
        t, v = data[k]
        if mint is None or t[0] < mint:
            mint = t[0]
        if maxt is None or t[-1] > maxt:
            maxt = t[-1]
        if colors_and_styles:
            c, s = colors_and_styles[k]
            plt.plot(t, v, label=str(k), color=c, ls=s)
        else:
            plt.plot(t, v, label=str(k))
    if mint is not None and maxt is not None:
        t = list(range(mint, maxt + 1))
    if title is not False:
        plt.title(title or f"{plot_kind._description} versus {xlabel.lower()}")
    if legend is None:
        legend = data and (len(data) <= LEGEND_LIMIT)
    if legend is True:
        plt.legend()
    elif legend:
        plt.legend(legend)
    plt.xlabel(xlabel)
    if mint is not None and maxt is not None:
        plt.xticks(_xticks(t))
    plt.ylabel(ylabel)
    plt.yticks(plot_kind.yticks())
    if limits:
        plt.ylim(limits)
    elif plot_kind._default_ylim:
        plt.ylim(plot_kind._default_ylim)
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
    return plt.gcf()

def _plot_ordering(df, data):
    options = sorted(df["option"].unique())
    if len(options) > len(PLOT_COLORS):
        return None
    utilities = sorted([n for n in df["utility"].unique()], reverse=True)
    if all(math.isclose(u, round(u)) for u in utilities):
        utilities = [int(u) for u in utilities]
    result = {}
    for o, c in zip(options, PLOT_COLORS):
        if o in data:
            result[o] = (c, PLOT_LINE_STYLES[0])
        else:
            line_styles = list(PLOT_LINE_STYLES)
            for u in utilities:
                if (s := f"{o}, {u}") in data:
                    if not line_styles:
                        return None
                    result[s] = (c, line_styles.pop(0))
    return result

def _xticks(times):
    def f(x):
        t0 = times[0]
        result = [t0]
        tx = math.ceil(t0 / x) * x
        if tx - t0 < 7:
            tx += x
        tend = times[-1]
        result += list(range(tx, tend + 1, x))
        if tend - result[-1] >= 7:
            result += [tend]
        else:
            result[-1] = tend
        return result
    try:
        if (n := len(times)) <= 6:
            return times
        elif n <= 15:
            return [times[i] for i in range(0, n, 2)]
        elif n <= 80:
            return f(10)
        elif n < 160:
            return f(20)
        elif n <= 240:
            return f(30)
        elif n <= 310:
            return f(40)
        elif n <= 400:
            return f(50)
        else:
            return None
    except Error:
        # shouldn't be possible, but just in case...
        return None



class Plot():

    def __init__(self, column, description, limit_range=False):
        self._column = column
        self._description = description
        self._default_ylim = (-0.05, 1.05) if limit_range else None

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        raise NotImplementedError()

    def yticks(self):
        return None


class ChoicePlot(Plot):

    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        counters = {}
        sums = defaultdict(int)
        times = set()
        for ch in data.choice.unique():
            d = data[data.choice==ch]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby(["iteration", "time"])
            # There's probably a better way to do this within Pandas, right?
            cnt = Counter([t for (i, t), ignore in grouped[["iteration", "time"]]])
            times.update(cnt.keys())
            for k in cnt:
                sums[k] += cnt[k]
            counters[ch] = cnt
        result = {}
        for ch, cnt in counters.items():
            if include and ch not in include:
                continue
            if exclude and ch in exclude:
                continue
            for k, v in cnt.items():
                cnt[k] /= sums[k]
            for t in times:
                if t not in cnt:
                    cnt[t] = 0
            result[ch] = tuple(zip(*sorted(cnt.items())))
        return result

    def yticks(self):
        return [0.0, 0.25, 0.5, 0.75, 1.0]


class OptionPlot(Plot):
    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        result = {}
        for opt in data.option.unique():
            if include and opt not in include:
                continue
            if exclude and opt in exclude:
                continue
            d = data[(data.option==opt)]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby("time")
            result[opt] = ([t for t, ignore in grouped["time"]],
                           grouped[self._column].mean())
        return result


class InstancePlot(Plot):
    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def get_data(self, data, include, exclude, min, max, earliest, latest):
        result = {}
        for opt, util in data[["option", "utility"]].drop_duplicates().itertuples(index=False, name=None):
            if include and opt not in include:
                continue
            if exclude and opt in exclude:
                continue
            if min and util < min:
                continue
            if max and util > max:
                continue
            d = data[(data.option==opt) & (data.utility==util)]
            if earliest is not None:
                d = d[d.time >= earliest]
            if latest is not None:
                d = d[d.time <= latest]
            grouped = d.groupby("time")
            result[f"{opt}, {util}"] = ([t for t, ignore in grouped["time"]],
                                        grouped[self._column].mean())
        return result


class RetrievalProbabilyPlot(InstancePlot):

    def __init__(self, column, description, limit_range=False):
        super().__init__(column, description, limit_range)

    def yticks(self):
        return [0.0, 0.25, 0.5, 0.75, 1.0]


