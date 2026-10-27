# Copyright 2014-2026 Carnegie Mellon University

"""Similarity helper functions used by PyIBL."""

from warnings import warn
def positive_linear_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled linearly by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
that gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_linear_similarity(1, 2)
0.5
>>> positive_linear_similarity(2, 1)
0.5
>>> positive_linear_similarity(1, 10)
0.09999999999999998
>>> positive_linear_similarity(10, 100)
0.09999999999999998
>>> positive_linear_similarity(1, 2000)
0.0004999999999999449
>>> positive_linear_similarity(1999, 2000)
0.9995
>>> positive_linear_similarity(1, 1)
1
>>> positive_linear_similarity(0.001, 0.002)
0.5
>>> positive_linear_similarity(10.001, 10.002)
0.9999000199960006
"""
    if x <= 0 or y <= 0:
        raise ValueError(f"the arguments, {x} and {y}, are not both positive")
    if x == y:
        return 1
    if x > y:
        x, y = y, x
    return 1 - (y - x) / y

def positive_quadratic_similarity(x, y):
    """Returns a similarity value of two positive :class:`Real` numbers, scaled quadratically by the larger of them.
If *x* and *y* are equal the value is one, and otherwise a positive float less than one
that gets smaller the greater the difference between *x* and *y*.

If either *x* or *y* is not positive a :exc:`ValueError` is raised.

>>> positive_quadratic_similarity(1, 2)
0.25
>>> positive_quadratic_similarity(2, 1)
0.25
>>> positive_quadratic_similarity(1, 10)
0.009999999999999995
>>> positive_quadratic_similarity(10, 100)
0.009999999999999995
>>> positive_quadratic_similarity(1, 2000)
2.4999999999994493e-07
>>> positive_quadratic_similarity(1999, 2000)
0.9990002500000001
>>> positive_quadratic_similarity(1, 1)
1
>>> positive_quadratic_similarity(0.001, 0.002)
0.25
>>> positive_quadratic_similarity(10.001, 10.002)
0.9998000499880025
"""
    return positive_linear_similarity(x, y)**2

def bounded_linear_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a linear scale between *minimum* and *maximum*.
The two arguments to the function returned should be :class:`Real` numbers between
*minimum* and *maximum*, inclusive. If the two arguments to the function returned are
equal they are maximally similar, and one is returned. If the absolute value of their
difference is as large as possible, they are maximally different, and zero is returned.
Otherwise a scaled value on a linear scale between these two extrema, measuring the
magnitude of the difference between the arguments to the returned function is used, a
value between zero and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_linear_similarity(-1, 1)
>>> f(0, 1)
0.5
>>> f(-0.1, 0.1)
0.9
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999999

    """
    if minimum >= maximum:
        raise ValueError(f"minimum, {minimum}, is not less than maximum, {maximum}")
    def _similarity(x, y):
        if x < minimum:
            warn(f"{x} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            x = minimum
        elif x > maximum:
            warn(f"{x} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            x = maximum
        if y < minimum:
            warn(f"{y} is less than {minimum}, so {minimum} is instead being used in computing similarity")
            y = minimum
        elif y > maximum:
            warn(f"{y} is greater than {maximum}, so {maximum} is instead being used in computing similarity")
            y = maximum
        return 1 - abs(x - y) / abs(maximum - minimum)
    return _similarity

def bounded_quadratic_similarity(minimum, maximum):
    """Returns a function of two arguments that returns a similarity value reflecting a quadratic scale between *minimum* and *maximum*.
Both arguments to the function returned should be :class:`Real` numbers between *minimum*
and *maximum*, inclusive. If the two arguments to the function returned are equal they are
maximally similar, and one is returned. If the absolute value of their difference is as
large as possible, they are maximally different, and zero is returned. Otherwise a scaled
value on a quadratic scale between these two extrema, measuring the magnitude of the
difference between the arguments to the returned function is used, a value between zero
and one being returned.

Raises a :exc:`ValueError` if either *minimum* or *maximum* is not a Real number, or if
*minimum* is not less than *maximum*.

When the returned function is called if either of its arguments is not a Real number a
:exc:`ValueError` is then raised. If either of those arguments is less than *minimum*,
or greater than *maximum*, a warning is issued, and either *minimum* or *maximum*,
respectively, is instead used as the argument's value.

>>> f = bounded_quadratic_similarity(-1, 1)
>>> f(0, 1)
0.25
>>> f(-0.1, 0.1)
0.81
>>> f(-1, 1)
0.0
>>> f(0, 0)
1.0
>>> sys.float_info.epsilon
2.220446049250313e-16
>>> f(0, _)
0.9999999999999998

    """
    f = bounded_linear_similarity(minimum, maximum)
    return lambda x, y: f(x, y)**2


# Local variables:
# fill-column: 90
# End:
