import abc
from typing import ValuesView

import numpy as np


def create_class_constructor(base_class, class_dict):
    """Creates a constructor function for subclasses of base_class.

    The constructor function returned takes in either None, a object that is an
    instance of base_class, or a dictionary config. If the argument is None or an
    instance of base_class, it is returned without modification. If it is a
    dictionary, the config should have two keys: name and kwargs. The name
    parameter is used to lookup the correct class from class_dict and the object is
    created using kwargs as parameters.

    Args:
        base_class (type|"callable"): If base_class is a type, it is used to verify
            the type of the object passed to the constructor. If base_class is the
            string "callable", the object passed to the constructor is simply checked
            to see if it's callable.
        class_dict: A dictionary of class names to callables that can be passed kwargs
            to create the necessary objects.
    """

    def constructor(object_or_config):
        if object_or_config is None:
            return None
        if base_class == "callable":
            if callable(object_or_config):
                return object_or_config
        elif isinstance(object_or_config, base_class):
            return object_or_config
        name = object_or_config["name"]
        kwargs = object_or_config["kwargs"]
        if name in class_dict:
            object_class = class_dict[name]
            return object_class(**kwargs)
        else:
            raise ValueError(f"{name} class not found")

    return constructor


class Schedule(abc.ABC):
    @abc.abstractmethod
    def get_value():
        """Returns the current value of the variable we are tracking"""
        pass

    @abc.abstractmethod
    def update():
        """Update the value of the variable we are tracking and return the updated value.
        The first call to update will return the initial value of the schedule."""
        pass


class LinearSchedule(Schedule):
    """Defines a linear schedule between two values over some number of steps.

    If updated more than the defined number of steps, the schedule stays at the
    end value.
    """

    def __init__(self, init_value, end_value, steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule.
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        steps = max(int(steps), 1)
        self._delta = (end_value - init_value) / steps
        self._end_value = end_value
        self._value = init_value - self._delta

    def get_value(self):
        return self._value

    def update(self):
        if self._value == self._end_value:
            return self._value

        self._value += self._delta

        # Check if value is over the end_value
        if ((self._value - self._end_value) > 0) == (self._delta > 0):
            self._value = self._end_value
        return self._value


class ConstantSchedule(Schedule):
    """Returns a constant value over the course of the schedule"""

    def __init__(self, value):
        """
        Args:
            value: the value to be returned.
        """
        self._value = value

    def get_value(self):
        return self._value

    def update(self):
        return self._value


class SwitchSchedule(Schedule):
    """Returns one value for the first part of the schedule. After the defined
    number of steps is reached, switches to returning a second value.
    """

    def __init__(self, off_value, on_value, steps):
        """
        Args:
            off_value: the value to be returned in the first part of the schedule.
            on_value: the value to be returned in the second part of the schedule.
            steps (int): the number of steps after which to switch from the off
                value to the on value.
        """

        self._steps = -1
        self._flip_step = steps
        self._off_value = off_value
        self._on_value = on_value

    def get_value(self):
        if self._steps < self._flip_step:
            return self._off_value
        else:
            return self._on_value

    def update(self):
        self._steps += 1
        value = self.get_value()
        return value


class DoublePeriodicSchedule(Schedule):
    """Returns off value for off period, then switches to returning on value for on
    period. Alternates between the two.
    """

    def __init__(self, off_value, on_value, off_period, on_period):
        """
        Args:
            on_value: the value to be returned for the on period.
            off_value: the value to be returned for the off period.
            on_period (int): the number of steps in the on period.
            off_period (int): the number of steps in the off period.
        """
        self._steps = -1
        self._off_period = off_period
        self._total_period = self._off_period + on_period
        self._off_value = off_value
        self._on_value = on_value

    def get_value(self):
        if (self._steps % self._total_period) < self._off_period:
            return self._off_value
        else:
            return self._on_value

    def update(self):
        self._steps += 1
        return self.get_value()


class PeriodicSchedule(DoublePeriodicSchedule):
    """Returns one value on the first step of each period of a predefined number of
    steps. Returns another value otherwise.
    """

    def __init__(self, off_value, on_value, period):
        """
        Args:
            on_value: the value to be returned on the first step of each period.
            off_value: the value to be returned for every other step in the period.
            period (int): the number of steps in the period.
        """
        super().__init__(off_value, on_value, period - 1, 1)


class LinearSchedule(Schedule):
    """Defines a linear schedule between two values over some number of steps.

    If updated more than the defined number of steps, the schedule stays at the
    end value.
    """

    def __init__(self, init_value, end_value, steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule.
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        steps = max(int(steps), 1)
        self._delta = (end_value - init_value) / steps
        self._end_value = end_value
        self._value = init_value - self._delta

    def get_value(self):
        return self._value

    def update(self):
        if self._value == self._end_value:
            return self._value

        self._value += self._delta

        # Check if value is over the end_value
        if ((self._value - self._end_value) > 0) == (self._delta > 0):
            self._value = self._end_value
        return self._value


class CosineSchedule(Schedule):
    """
    Cosine schedule
    """

    def __init__(self, init_value, end_value, steps):
        """
        Args:
            init_value (Union[int, float]): starting value for schedule.
            end_value (Union[int, float]): end value for schedule.
            steps (int): Number of steps for schedule. Should be positive.
        """
        self._init_value = init_value
        self._end_value = end_value
        self._steps = max(int(steps), 1)
        self._current_step = 0

    def get_value(self):
        a = 0.5 * (self._init_value - self._end_value)
        b = 0.5 * (self._init_value + self._end_value)
        t = min(self._current_step, self._steps) / self._steps
        return a * np.cos(np.pi * t) + b

    def update(self):
        value = self.get_value()
        self._current_step += 1
        return value


get_schedule = create_class_constructor(
    Schedule,
    {
        "LinearSchedule": LinearSchedule,
        "ConstantSchedule": ConstantSchedule,
        "SwitchSchedule": SwitchSchedule,
        "PeriodicSchedule": PeriodicSchedule,
        "DoublePeriodicSchedule": DoublePeriodicSchedule,
        "CosineShedule": CosineSchedule,
    },
)
