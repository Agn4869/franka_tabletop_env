"""Finite, observation-driven pick/place execution. Planner attachments are NOT physics.

The host calls command_for_next_interval(obs), applies the returned articulation
controller target, advances exactly one interval, then observe_transition(next_obs).
The executor never advances physics, sets poses, or changes robot joint state.
"""
from __future__ import annotations

import time
from dataclasses import replace
import numpy as np

from .contracts import (AngularGripper, ControlCommand, MotionPlannerAdapter,
                        PickPlaceTask, TaskProgress, WorldObservation)
from .geometry import (base_frame_target, grasp_end_effector_pose, inverse_transform,
                       pose_error, pregrasp_pose)
from .trajectory import JointLimits, JointTrajectory


class ExecutionStopped(RuntimeError):
    def __init__(self, progress: TaskProgress):
        self.progress = progress
        super().__init__(progress.failure_reason or "execution already finished")


class PickPlaceExecutor:
    STAGES = ("PREGRASP", "APPROACH", "CLOSE", "LIFT", "TRANSFER", "PLACE", "RELEASE", "SETTLE")

    def __init__(self, planner: MotionPlannerAdapter, gripper: AngularGripper, limits: JointLimits):
        self.planner, self.gripper, self.limits = planner, gripper, limits
        self.task = None
        self._progress = TaskProgress(False, False, "IDLE")
        self._cleaned = True
        self._planning_events = []

    @property
    def progress(self):
        events = tuple(dict(event) for event in self._planning_events)
        return replace(self._progress, planning_wall_time_s=sum(event["wall_time_s"] for event in events),
                       planning_attempts=len(events), planning_events=events)

    def start(self, task_spec: PickPlaceTask):
        if self.task is not None and not self.progress.done:
            raise RuntimeError("an episode is already running")
        self.task = task_spec
        self._planning_events = []
        self._candidate_index = 0
        self._plan = None
        self._pending = None
        self._episode_start = None
        self._stage_start = None
        self._stable_since = None
        self._relative_grasp = None
        self._grip_reference = None
        self._initial_object_z = None
        self._target = None
        self._grasp_target = None
        self._attached = False
        self._released = False
        self._replans = 0
        self._last_observed_key = None
        self._cleaned = False
        self._progress = TaskProgress(False, False, "PREGRASP", candidate_id=task_spec.candidates[0].candidate_id)

    def _cleanup(self):
        if not self._cleaned:
            # Mark first, so an exception doesn't cause an unbounded retry loop.
            self._cleaned = True
            self.planner.cleanup()
            self._attached = False

    def _end(self, success: bool, reason=None):
        try:
            self._cleanup()
        except Exception as error:
            success = False
            reason = f"{reason + ';' if reason else ''}planner_cleanup_failed:{type(error).__name__}:{error}"
        self._plan = None
        self._pending = None
        self._progress = TaskProgress(True, success, self.progress.stage, reason, self.progress.candidate_id)
        return self.progress

    def abort(self, reason="aborted"):
        if self.task is None:
            raise RuntimeError("executor has not started")
        if not self.progress.done:
            self._end(False, reason)
        return self.progress

    def close(self):
        if self.task is not None and not self.progress.done:
            self.abort("executor_closed")

    def _stage(self, stage, observation):
        self._plan = None
        self._target = None
        self._stage_start = observation.sim_time
        self._stable_since = None
        self._replans = 0
        self._progress = TaskProgress(False, False, stage, candidate_id=self.task.candidates[self._candidate_index].candidate_id)

    def _check_observation(self, obs):
        if set(obs.joint_names) != set(self.limits.joint_names):
            raise ValueError("observed and controlled joint sets differ")
        lower, upper, _, _ = self.limits.ordered(obs.joint_names)
        if np.any(obs.joint_positions < lower - 1e-6) or np.any(obs.joint_positions > upper + 1e-6):
            raise ValueError("observed_joint_limit")

    def _ordered_positions(self, obs):
        indices = [obs.joint_names.index(name) for name in self.limits.joint_names]
        return obs.joint_positions[indices]

    def _make_target(self, obs):
        stage = self.progress.stage
        if stage == "PREGRASP":
            candidate = self.task.candidates[self._candidate_index]
            self._grasp_target = grasp_end_effector_pose(obs.T_W_O, candidate.T_O_G, self.task.T_E_G)
            return pregrasp_pose(obs.T_W_O, candidate.T_O_G, self.task.T_E_G,
                                 candidate.approach_direction_G, candidate.pregrasp_distance_m)
        if stage == "APPROACH":
            return self._grasp_target.copy()
        if stage == "LIFT":
            result = obs.T_W_E.copy()
            result[2, 3] += self.task.lift_height_m
            return result
        if stage in ("TRANSFER", "PLACE"):
            object_target = self.task.T_W_O_place.copy()
            if stage == "TRANSFER":
                object_target[2, 3] += self.task.lift_height_m
            return object_target @ inverse_transform(self._relative_grasp)
        if stage == "RELEASE" and self._released:
            result = obs.T_W_E.copy()
            result[2, 3] += self.task.retreat_distance_m
            return result
        return None

    def _next_candidate(self, obs):
        count = min(len(self.task.candidates), self.task.max_candidates)
        if self._candidate_index + 1 >= count:
            return False
        self._candidate_index += 1
        self._grasp_target = None
        self._stage("PREGRASP", obs)
        return True

    def _build_plan(self, obs):
        """Bounded candidates; failures never become a fabricated trajectory."""
        while True:
            try:
                if self._target is None:
                    self._target = self._make_target(obs)
                if self._target is None:
                    return
                self._plan = self._attempt_plan(obs)
                self._plan_started = obs.sim_time
                self._plan_observed_world_version = obs.world_version
                return
            except Exception as error:
                # Only PREGRASP planning failures may choose another annotation.
                # Dependency/runtime failures are visible; none trigger a fake planner.
                if isinstance(error, ValueError) and self.progress.stage == "PREGRASP" and self._next_candidate(obs):
                    continue
                self._end(False, f"planning_failed:{type(error).__name__}:{error}")
                raise ExecutionStopped(self.progress) from error

    def _attempt_plan(self, obs):
        started = time.monotonic()
        duration = None
        event = {"stage": self.progress.stage, "candidate_id": self.progress.candidate_id,
                 "physics_step": obs.step_index, "sim_time": obs.sim_time,
                 "started_wall_time": time.time(), "success": False, "failure_reason": None}
        try:
            set_stage = getattr(self.planner, "set_stage", None)
            if set_stage is not None:
                set_stage(self.progress.stage, self.task.object_id)
            version = self.planner.sync_world(obs)
            plan = self.planner.plan(self.limits.joint_names, self._ordered_positions(obs),
                                     base_frame_target(obs.T_W_B, self._target), version)
            duration = max(0.0, time.monotonic() - started)
            if duration > self.task.planner_timeout_s:
                raise TimeoutError("planner_wall_timeout")
            if not isinstance(plan, JointTrajectory) or plan.world_version != version:
                raise ValueError("planner_world_version_mismatch")
            plan = plan.reordered(self.limits.joint_names).validate_limits(self.limits)
            if np.max(np.abs(plan.positions[0] - self._ordered_positions(obs))) > self.task.joint_tolerance_rad:
                raise ValueError("trajectory_discontinuous_start")
            event.update(success=True, plan_id=plan.plan_id, world_version=version)
            return plan
        except BaseException as error:
            event["failure_reason"] = f"{type(error).__name__}:{error}"
            raise
        finally:
            event["wall_time_s"] = duration if duration is not None else max(0.0, time.monotonic() - started)
            self._planning_events.append(event)

    def _safety(self, obs):
        if obs.unexpected_collision:
            return self._end(False, "unexpected_collision")
        if obs.sim_time - self._episode_start > self.task.episode_timeout_s:
            return self._end(False, "episode_timeout")
        if obs.sim_time - self._stage_start > self.task.stage_timeout_s:
            if self.progress.stage == "PREGRASP" and self._next_candidate(obs):
                return self.progress
            return self._end(False, f"stage_timeout:{self.progress.stage.lower()}")
        if self._attached and self.progress.stage in ("LIFT", "TRANSFER", "PLACE"):
            if not obs.contact_confirmed:
                return self._end(False, f"object_dropped_during_{self.progress.stage.lower()}")
            relative = inverse_transform(obs.T_W_E) @ obs.T_W_O
            distance, angle = pose_error(relative, self._relative_grasp)
            if distance > self.task.slip_tolerance_m or angle > self.task.slip_tolerance_rad:
                return self._end(False, "object_slipped")
        return self.progress

    def command_for_next_interval(self, observation: WorldObservation):
        obs = observation
        if self.task is None:
            raise RuntimeError("call start first")
        if self.progress.done:
            raise ExecutionStopped(self.progress)
        if self._pending is not None:
            raise RuntimeError("observe_transition must consume the preceding command")
        if self._last_observed_key is not None and self._last_observed_key != (obs.step_index, obs.sim_time):
            self._end(False, "stale_observation_for_command")
            raise ExecutionStopped(self.progress)
        try:
            self._check_observation(obs)
        except ValueError as error:
            self._end(False, str(error))
            raise ExecutionStopped(self.progress) from error
        if self._episode_start is None:
            self._episode_start = self._stage_start = obs.sim_time
            self._initial_object_z = float(obs.T_W_O[2, 3])
        if self._safety(obs).done:
            raise ExecutionStopped(self.progress)
        moving = self.progress.stage in ("PREGRASP", "APPROACH", "LIFT", "TRANSFER", "PLACE") or (
            self.progress.stage == "RELEASE" and self._released)
        if moving:
            if self._plan is not None and obs.world_version != self._plan_observed_world_version:
                self._replans += 1
                if self._replans > self.task.max_replans:
                    self._end(False, "world_changed_replan_budget_exhausted")
                    raise ExecutionStopped(self.progress)
                self._plan = None
                if self.progress.stage == "PREGRASP":
                    self._target = None
                elif self.progress.stage == "APPROACH":
                    candidate = self.task.candidates[self._candidate_index]
                    self._grasp_target = grasp_end_effector_pose(obs.T_W_O, candidate.T_O_G, self.task.T_E_G)
                    self._target = self._grasp_target.copy()
            if self._plan is None:
                self._build_plan(obs)
            elapsed = obs.sim_time - self._plan_started + self.task.control_dt
            positions = self._plan.sample(elapsed)
            plan_id, version = self._plan.plan_id, self._plan.world_version
        else:
            positions = self._ordered_positions(obs)
            plan_id, version = "", obs.world_version
        ratio = 1.0 if self.progress.stage in ("PREGRASP", "APPROACH", "RELEASE", "SETTLE") else 0.0
        command = ControlCommand(self.limits.joint_names, positions, ratio, self.gripper.angle(ratio),
                                 self.progress.stage, version, plan_id, candidate_id=self.progress.candidate_id)
        self._pending = (command, obs.step_index, obs.sim_time)
        return command

    def _at_target(self, obs):
        if self._plan is None or obs.sim_time - self._plan_started < self._plan.duration - 1e-8:
            return False
        joint_error = np.max(np.abs(self._ordered_positions(obs) - self._plan.positions[-1]))
        translation, rotation = pose_error(obs.T_W_E, self._target)
        return (joint_error <= self.task.joint_tolerance_rad and translation <= self.task.pose_tolerance_m
                and rotation <= self.task.rotation_tolerance_rad)

    def _stable(self, predicate, obs, seconds):
        if not predicate:
            self._stable_since = None
            return False
        if self._stable_since is None:
            self._stable_since = obs.sim_time
        return obs.sim_time - self._stable_since >= seconds - 1e-8

    def _inside_bin(self, obs):
        # Require the complete oriented object's conservative world AABB, not
        # merely its centre, inside the configured usable (not outer) volume.
        radius = np.abs(obs.T_W_O[:3, :3]) @ self.task.object_half_extents
        position = obs.T_W_O[:3, 3]
        epsilon = self.task.placement_tolerance_m
        return bool(np.all(position - radius >= self.task.bin_min_W - epsilon)
                    and np.all(position + radius <= self.task.bin_max_W + epsilon))

    def observe_transition(self, observation: WorldObservation):
        if self.progress.done:
            return self.progress
        if self._pending is None:
            raise RuntimeError("no outstanding controller command")
        command, old_step, old_time = self._pending
        self._pending = None
        obs = observation
        try:
            self._check_observation(obs)
        except ValueError as error:
            return self._end(False, str(error))
        if obs.step_index != old_step + self.task.physics_steps_per_interval or not np.isclose(
            obs.sim_time - old_time, self.task.control_dt, atol=1e-7, rtol=1e-6
        ):
            return self._end(False, "action_observation_time_mismatch")
        self._last_observed_key = (obs.step_index, obs.sim_time)
        if self._safety(obs).done:
            return self.progress
        if command.stage != self.progress.stage:
            # A timed-out PREGRASP selected a new candidate; don't apply the old
            # candidate's transition to the new candidate.
            return self.progress
        if np.max(np.abs(self._ordered_positions(obs) - command.joint_positions)) > self.task.tracking_tolerance_rad:
            return self._end(False, "joint_tracking_error")
        if self._plan is not None and obs.world_version != self._plan_observed_world_version:
            # No phase transition on an invalidated world; the next command
            # must explicitly replan before it can advance.
            return self.progress
        stage = self.progress.stage
        if stage == "PREGRASP" and self._at_target(obs):
            self._stage("APPROACH", obs)
        elif stage == "APPROACH" and self._at_target(obs):
            self._stage("CLOSE", obs)
            self._grip_reference = inverse_transform(obs.T_W_E) @ obs.T_W_O
        elif stage == "CLOSE":
            relative = inverse_transform(obs.T_W_E) @ obs.T_W_O
            translation, rotation = pose_error(relative, self._grip_reference)
            # A real object stops angular fingers before the empty-gripper
            # closed angle. Require an issued CLOSE command and sustained
            # bilateral contact, not a guessed remaining aperture threshold.
            held = (command.gripper_open_ratio == 0.0 and obs.contact_confirmed
                    and translation <= self.task.slip_tolerance_m and rotation <= self.task.slip_tolerance_rad)
            if self._stable(held, obs, self.task.grip_hold_s):
                try:
                    self._relative_grasp = relative.copy()
                    self.planner.attach(self.task.object_id, relative)
                    self._attached = True
                    self._stage("LIFT", obs)
                except Exception as error:
                    return self._end(False, f"planner_attach_failed:{error}")
        elif stage == "LIFT":
            lifted = obs.T_W_O[2, 3] - self._initial_object_z >= self.task.min_lift_m
            if self._stable(lifted and self._at_target(obs), obs, self.task.lift_hold_s):
                self._stage("TRANSFER", obs)
        elif stage == "TRANSFER" and self._at_target(obs):
            self._stage("PLACE", obs)
        elif stage == "PLACE" and self._at_target(obs):
            if not self._inside_bin(obs):
                return self._end(False, "object_outside_placement_volume")
            self._stage("RELEASE", obs)
        elif stage == "RELEASE":
            if not self._released and obs.gripper_open_ratio >= 0.8 and not obs.contact_confirmed:
                try:
                    self.planner.detach(self.task.object_id, obs.T_W_O)
                    self._attached = False
                    self._released = True
                    self._plan = self._target = None
                except Exception as error:
                    return self._end(False, f"planner_detach_failed:{error}")
            elif self._released and self._at_target(obs):
                self._stage("SETTLE", obs)
        elif stage == "SETTLE":
            stable = (self._inside_bin(obs) and not obs.contact_confirmed and obs.gripper_open_ratio >= 0.8
                      and np.linalg.norm(obs.object_linear_velocity) <= self.task.settle_linear_speed
                      and np.linalg.norm(obs.object_angular_velocity) <= self.task.settle_angular_speed)
            if self._stable(stable, obs, self.task.settle_hold_s):
                return self._end(True)
        return self.progress
