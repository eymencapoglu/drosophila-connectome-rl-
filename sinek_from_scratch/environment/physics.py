import math
import numpy as np

from config import (
    GRAVITY,
    DT,
    AIR_DRAG,
    ANGULAR_DRAG,
    MAX_SPEED,
    MAX_ANGULAR_SPEED,
    FLY_RADIUS,
    FLY_MASS,
    MAX_LIFT,
    MAX_THRUST,
    MAX_TORQUE,
)


class FlyPhysics:

    def __init__(self):

        self.reset()

    def reset(
        self,
        x=20.0,
        y=50.0,
        heading=0.0,
    ):

        self.x = float(x)
        self.y = float(y)

        self.vx = 0.0
        self.vy = 0.0

        self.ax = 0.0
        self.ay = 0.0

        self.heading = float(heading)

        self.angular_velocity = 0.0

        self.left_wing = 0.0
        self.right_wing = 0.0

        self.energy = 1.0

        self.on_ground = False

    # ========================================================
    # WING
    # ========================================================

    def apply_wings(
        self,
        left,
        right,
    ):

        self.left_wing = float(np.clip(left, -1.0, 1.0))
        self.right_wing = float(np.clip(right, -1.0, 1.0))

        average_wing = (
            self.left_wing +
            self.right_wing
        ) / 2.0

        difference = (
            self.right_wing -
            self.left_wing
        )

        # ----------------------------------------------------
        # Lift
        # ----------------------------------------------------

        lift = average_wing * MAX_LIFT

        # ----------------------------------------------------
        # Forward thrust
        # ----------------------------------------------------

        thrust = (
            abs(average_wing) *
            MAX_THRUST
        )

        # ----------------------------------------------------
        # Direction
        # ----------------------------------------------------

        hx = math.cos(self.heading)
        hy = math.sin(self.heading)

        # Forward thrust
        self.ax += (
            hx *
            thrust /
            FLY_MASS
        )

        self.ay += (
            hy *
            thrust /
            FLY_MASS
        )

        # Lift
        self.ay += lift / FLY_MASS

        # ----------------------------------------------------
        # Turning
        # ----------------------------------------------------

        torque = (
            difference *
            MAX_TORQUE
        )

        self.angular_velocity += (
            torque *
            DT
        )

        self.angular_velocity = np.clip(
            self.angular_velocity,
            -MAX_ANGULAR_SPEED,
            MAX_ANGULAR_SPEED,
        )

        # ----------------------------------------------------
        # Energy
        # ----------------------------------------------------

        energy_cost = (
            abs(self.left_wing) +
            abs(self.right_wing)
        ) * 0.0005

        self.energy -= energy_cost

        self.energy = max(
            0.0,
            self.energy
        )

    # ========================================================
    # PHYSICS STEP
    # ========================================================

    def step(
        self,
        world_width,
        world_height,
        wind_x=0.0,
        wind_y=0.0,
    ):

        # Gravity
        self.ay -= GRAVITY

        # Wind
        self.ax += wind_x
        self.ay += wind_y

        # Acceleration → velocity
        self.vx += self.ax * DT
        self.vy += self.ay * DT

        # Drag
        self.vx *= AIR_DRAG
        self.vy *= AIR_DRAG

        # Speed limit
        speed = math.sqrt(
            self.vx ** 2 +
            self.vy ** 2
        )

        if speed > MAX_SPEED:

            scale = MAX_SPEED / speed

            self.vx *= scale
            self.vy *= scale

        # Position
        self.x += self.vx * DT
        self.y += self.vy * DT

        # Rotation
        self.heading += (
            self.angular_velocity *
            DT
        )

        self.heading = (
            self.heading +
            math.pi
        ) % (
            2 * math.pi
        ) - math.pi

        # Angular drag
        self.angular_velocity *= ANGULAR_DRAG

        # ----------------------------------------------------
        # WORLD BOUNDARIES
        # ----------------------------------------------------

        self.on_ground = False

        if self.x < FLY_RADIUS:

            self.x = FLY_RADIUS
            self.vx *= -0.4

        if self.x > world_width - FLY_RADIUS:

            self.x = (
                world_width -
                FLY_RADIUS
            )

            self.vx *= -0.4

        if self.y < FLY_RADIUS:

            self.y = FLY_RADIUS

            self.vy *= -0.2

            self.on_ground = True

        if self.y > world_height - FLY_RADIUS:

            self.y = (
                world_height -
                FLY_RADIUS
            )

            self.vy *= -0.4

        # Reset forces
        self.ax = 0.0
        self.ay = 0.0

    # ========================================================
    # STATE
    # ========================================================

    def state(self):

        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "heading": self.heading,
            "angular_velocity": self.angular_velocity,
            "left_wing": self.left_wing,
            "right_wing": self.right_wing,
            "energy": self.energy,
            "on_ground": self.on_ground,
        }