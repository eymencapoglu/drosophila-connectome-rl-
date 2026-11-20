import math
import random

import numpy as np

from config import (
    WORLD_WIDTH,
    WORLD_HEIGHT,
    FLY_RADIUS,
    SUGAR_RADIUS,
    SUGAR_REWARD,
    MAX_STEPS_PER_EPISODE,
    WIND_VARIATION,
    DISTANCE_REWARD_SCALE,
    TIME_PENALTY,
    ENERGY_COST_SCALE,
    FLIGHT_REWARD,
)

from environment.physics import FlyPhysics

from environment.sensors import (
    FlyVision,
    Proprioception,
)


# ============================================================
# FLY ENVIRONMENT
# ============================================================

class FlyEnvironment:

    def __init__(self):

        self.width = WORLD_WIDTH

        self.height = WORLD_HEIGHT

        self.fly = FlyPhysics()

        self.vision = FlyVision()

        self.proprioception = (
            Proprioception()
        )

        self.sugar_x = 0.0

        self.sugar_y = 0.0

        self.step_count = 0

        self.previous_distance = None

        self.wind_x = 0.0

        self.wind_y = 0.0

        self.last_observation = None

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.step_count = 0

        # ----------------------------------------------------
        # Random fly position
        # ----------------------------------------------------

        self.fly.reset(
            x=random.uniform(
                10.0,
                self.width - 10.0
            ),

            y=random.uniform(
                25.0,
                self.height - 15.0
            ),

            heading=random.uniform(
                -math.pi,
                math.pi
            )
        )

        # ----------------------------------------------------
        # Random sugar
        # ----------------------------------------------------

        self.sugar_x = random.uniform(
            10.0,
            self.width - 10.0
        )

        self.sugar_y = random.uniform(
            10.0,
            self.height - 10.0
        )

        # ----------------------------------------------------
        # Prevent extremely easy initial state
        # ----------------------------------------------------

        for _ in range(100):

            distance = (
                self.distance_to_sugar()
            )

            if distance >= 20.0:

                break

            self.sugar_x = random.uniform(
                10.0,
                self.width - 10.0
            )

            self.sugar_y = random.uniform(
                10.0,
                self.height - 10.0
            )

        # ----------------------------------------------------
        # Wind
        # ----------------------------------------------------

        self.wind_x = random.uniform(
            -WIND_VARIATION,
            WIND_VARIATION
        )

        self.wind_y = random.uniform(
            -WIND_VARIATION * 0.25,
            WIND_VARIATION * 0.25
        )

        self.previous_distance = (
            self.distance_to_sugar()
        )

        observation = (
            self.get_observation()
        )

        self.last_observation = observation

        return observation

    # ========================================================
    # DISTANCE
    # ========================================================

    def distance_to_sugar(self):

        dx = (
            self.sugar_x -
            self.fly.x
        )

        dy = (
            self.sugar_y -
            self.fly.y
        )

        return math.sqrt(
            dx * dx +
            dy * dy
        )

    # ========================================================
    # ACTION
    # ========================================================

    def action_to_wings(
        self,
        action
    ):

        actions = {

            0: (0.0, 0.0),

            1: (0.25, 0.25),

            2: (0.50, 0.50),

            3: (0.80, 0.80),

            4: (1.00, 1.00),

            5: (0.20, 0.80),

            6: (0.80, 0.20),

            7: (-0.20, 0.80),

            8: (0.80, -0.20),
        }

        return actions.get(
            int(action),
            (0.0, 0.0)
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action
    ):

        self.step_count += 1

        left, right = (
            self.action_to_wings(
                action
            )
        )

        # ----------------------------------------------------
        # APPLY MOTOR COMMAND
        # ----------------------------------------------------

        self.fly.apply_wings(
            left,
            right
        )

        # ----------------------------------------------------
        # PHYSICS
        # ----------------------------------------------------

        self.fly.step(
            self.width,
            self.height,
            self.wind_x,
            self.wind_y
        )

        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------

        distance = (
            self.distance_to_sugar()
        )

        distance_progress = (
            self.previous_distance -
            distance
        )

        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        reward = 0.0

        # Progress toward sugar.
        reward += (
            distance_progress *
            DISTANCE_REWARD_SCALE
        )

        # Time has a small cost.
        reward -= TIME_PENALTY

        # Flight reward.
        if not self.fly.on_ground:

            reward += FLIGHT_REWARD

        # Energy expenditure.
        energy_cost = (
            abs(left) +
            abs(right)
        )

        reward -= (
            energy_cost *
            ENERGY_COST_SCALE
        )

        terminated = False

        # ----------------------------------------------------
        # SUGAR
        # ----------------------------------------------------

        if distance <= (
            FLY_RADIUS +
            SUGAR_RADIUS
        ):

            reward += SUGAR_REWARD

            terminated = True

        # ----------------------------------------------------
        # TIME LIMIT
        # ----------------------------------------------------

        truncated = (
            self.step_count >=
            MAX_STEPS_PER_EPISODE
        )

        self.previous_distance = (
            distance
        )

        observation = (
            self.get_observation()
        )

        self.last_observation = (
            observation
        )

        info = {

            "distance":
                distance,

            "x":
                self.fly.x,

            "y":
                self.fly.y,

            "vx":
                self.fly.vx,

            "vy":
                self.fly.vy,

            "ax":
                self.fly.ax,

            "ay":
                self.fly.ay,

            "heading":
                self.fly.heading,

            "angular_velocity":
                self.fly.angular_velocity,

            "left_wing":
                self.fly.left_wing,

            "right_wing":
                self.fly.right_wing,

            "wind_x":
                self.wind_x,

            "wind_y":
                self.wind_y,

            "energy":
                self.fly.energy,

            "success":
                terminated,

            "sugar_x":
                self.sugar_x,

            "sugar_y":
                self.sugar_y,
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ========================================================
    # OBSERVATION
    # ========================================================

    def get_observation(self):

        vision = (
            self.vision.observe(
                self.fly,
                self.sugar_x,
                self.sugar_y,
                self.width,
                self.height
            )
        )

        proprio = (
            self.proprioception.observe(
                self.fly
            )
        )

        return {

            "vision":
                vision.astype(
                    np.float32
                ),

            "proprioception":
                proprio.astype(
                    np.float32
                ),
        }

    # ========================================================
    # DEBUG STATE
    # ========================================================

    def debug_state(self):

        return {

            "fly":
                self.fly.state(),

            "sugar_x":
                self.sugar_x,

            "sugar_y":
                self.sugar_y,

            "wind_x":
                self.wind_x,

            "wind_y":
                self.wind_y,

            "distance":
                self.distance_to_sugar(),

            "step":
                self.step_count,
        }