"""Isaac Sim Franka PickPlaceController backend.

This backend owns only controller construction and command generation. The host
still applies actions, advances PhysX, captures observations and evaluates the
physical task result.
"""
from __future__ import annotations

import numpy as np


class FrankaPickPlaceBackend:
    name = "franka_pickplace"

    def __init__(self, session, robot):
        self.session = session
        self.robot = robot
        self.controller = None
        self.action_controller = robot.get_articulation_controller()

    def start(self, *, end_effector_initial_height: float, events_dt, gripper):
        from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
        self.controller = PickPlaceController(
            name="shared_franka_pick_place",
            gripper=gripper,
            robot_articulation=self.robot,
            end_effector_initial_height=float(end_effector_initial_height),
            events_dt=list(events_dt),
        )
        return self.controller

    def command(self, *, picking_position, placing_position, current_joint_positions,
                end_effector_orientation, end_effector_offset=None):
        if self.controller is None:
            raise RuntimeError("Franka backend must be started before command")
        return self.controller.forward(
            picking_position=np.asarray(picking_position, dtype=float),
            placing_position=np.asarray(placing_position, dtype=float),
            current_joint_positions=current_joint_positions,
            end_effector_orientation=np.asarray(end_effector_orientation, dtype=float),
            end_effector_offset=(np.zeros(3) if end_effector_offset is None else
                                 np.asarray(end_effector_offset, dtype=float)),
        )

    def is_done(self):
        return bool(self.controller is not None and self.controller.is_done())

    def event(self):
        return int(self.controller.get_current_event()) if self.controller is not None else -1

    def cleanup(self):
        self.controller = None
        self.action_controller = None
