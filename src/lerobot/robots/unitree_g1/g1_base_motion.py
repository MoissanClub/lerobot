# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.
"""Explicitly activated stock high-level locomotion; never releases stock mode."""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np


class G1BaseMotion:
    def __init__(
        self,
        feedback: Callable[[], dict],
        *,
        max_speed: float = 0.1,
        max_yaw: float = 0.1,
        timeout: float = 0.2,
        client: Any = None,
        domain_id: int = 0,
    ):
        if not all(np.isfinite(v) and 0 < v <= 0.3 for v in (max_speed, max_yaw, timeout)):
            raise ValueError("Reviewed base limits must be positive and at most 0.3")
        self.feedback, self.client = feedback, client
        self.limits = np.array([max_speed, max_speed, max_yaw])
        self.timeout = timeout
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.target = np.zeros(3)
        self.updated = None
        self.thread = None
        self.fault = None
        self.damp_requested = False
        if type(domain_id) is not int or domain_id < 0:
            raise ValueError("Invalid DDS domain")
        self.domain_id = domain_id
        self.lease = None

    def activate(self) -> None:
        if self.thread is not None or self.stop.is_set():
            raise RuntimeError("Base session is single-use")
        self.feedback()
        import fcntl

        self.lease = open(f"/tmp/lerobot-g1-base-{self.domain_id}.lock", "a")  # noqa: SIM115
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lease.close()
            self.lease = None
            raise
        if self.client is None:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

            self.client = LocoClient()
            self.client.SetTimeout(0.1)
            self.client.Init()
        self.updated = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name="g1-base-watchdog")
        self.thread.start()

    def send(self, velocity: np.ndarray | list[float]) -> None:
        if self.fault or self.stop.is_set() or self.thread is None:
            raise RuntimeError(self.fault or "Base control not active")
        values = np.asarray(velocity, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            self.stop.set()
            raise ValueError("Invalid base velocity")
        with self.lock:
            self.target = np.clip(values, -self.limits, self.limits)
            self.updated = time.monotonic()

    @staticmethod
    def _check(code: int) -> None:
        if code != 0:
            raise RuntimeError(f"Locomotion RPC failed: {code}")

    def _run(self) -> None:
        try:
            while not self.stop.wait(0.02):
                self.feedback()
                with self.lock:
                    if time.monotonic() - self.updated > self.timeout:
                        raise RuntimeError("Base input timeout")
                    velocity = self.target.copy()
                # Same service as Move, but retain return code and bound command duration.
                self._check(self.client.SetVelocity(*velocity.tolist(), self.timeout))
        except Exception as exc:
            self.fault = str(exc)
        finally:
            self.stop.set()
            try:
                self._check(self.client.SetVelocity(0.0, 0.0, 0.0, self.timeout))
                if self.damp_requested:
                    self._check(self.client.SetFsmId(1))
            except Exception as exc:
                self.fault = f"{self.fault or 'Shutdown'}; stop RPC failed: {exc}"

    def close(self, *, damp: bool = False) -> None:
        self.damp_requested = damp
        self.stop.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(2)
            if self.thread.is_alive():
                raise RuntimeError("Base RPC thread did not stop; independent operator stop required")
        if self.lease is not None:
            self.lease.close()
            self.lease = None
        if self.fault:
            raise RuntimeError(self.fault)
