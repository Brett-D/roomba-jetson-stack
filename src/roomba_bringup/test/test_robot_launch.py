#!/usr/bin/env python3
"""robot.launch.py's driver and battery learner, decided without starting anything.

driver and battery_learner must be read the way IfCondition reads them (true/false/1/0, in any
case), and together: driver:=True once remapped the driver's battery_state to roomba/battery_raw
while the learner's own condition only accepted 'true', leaving /battery_state without a publisher.

  python3 -m pytest src/roomba_bringup/test/test_robot_launch.py    (or run this file)
Needs a sourced ROS 2 (launch, launch_ros); nothing needs to be built.
"""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import OpaqueFunction
from launch.conditions import InvalidConditionExpressionError
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

LAUNCH = Path(__file__).resolve().parent.parent / 'launch' / 'robot.launch.py'
REMAP = ('battery_state', 'roomba/battery_raw')


def load():
    spec = importlib.util.spec_from_file_location('robot_launch', LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # generate_launch_description looks up its own share directory: use the source tree.
    module.get_package_share_directory = lambda package: str(LAUNCH.parent.parent)
    return module


def driver_actions(driver, learner, rust='true'):
    """(package, executable, remappings) of each node _driver starts with these launch arguments."""
    context = LaunchContext()
    context.launch_configurations.update({
        'driver': driver, 'battery_learner': learner, 'rust_nodes': rust,
        'driver_params': 'driver.yaml', 'battery_learner_params': 'battery_learner.yaml'})
    nodes = load()._driver(context)
    assert all(isinstance(n, Node) for n in nodes)
    return [(n.node_package, n.node_executable,
             [(perform_substitutions(context, a), perform_substitutions(context, b)) for a, b in n._Node__remappings or []])
            for n in nodes]


def test_learner_on_in_any_spelling():
    for driver in ('true', 'True', 'TRUE', '1'):
        for learner in ('true', 'True', '1'):
            assert driver_actions(driver, learner) == [
                ('roomba_driver_rs', 'roomba_driver', [REMAP]),
                ('roomba_rs', 'battery_learner', []),
            ], (driver, learner)


def test_learner_off_leaves_the_driver_as_before():
    for learner in ('false', 'False', '0'):
        assert driver_actions('True', learner) == [('roomba_driver_rs', 'roomba_driver', [])], learner
    assert driver_actions('true', 'false', rust='false') == [('roomba_driver', 'roomba_driver', [])]


def test_no_driver_no_learner():
    for driver in ('false', 'False', '0'):
        assert driver_actions(driver, 'true') == [], driver


def test_bad_value_is_an_error_as_with_ifcondition():
    for driver, learner in (('true', 'yes'), ('on', 'true')):
        try:
            driver_actions(driver, learner)
        except InvalidConditionExpressionError:
            continue
        raise AssertionError(f'driver:={driver} battery_learner:={learner} was accepted')


def test_learner_starts_only_from_the_driver_action():
    module = load()
    entities = module.generate_launch_description().entities
    assert not [e for e in entities if isinstance(e, Node) and e._Node__node_name == 'battery_learner']
    assert [e for e in entities if isinstance(e, OpaqueFunction) and e._OpaqueFunction__function.__name__ == '_driver']


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
