import math
import numpy as np

from config import (
    VISION_WIDTH,
    VISION_HEIGHT,
    VISION_RANGE,
    FOV_DEGREES,
)


# ============================================================
# EGOCENTRIC COMPOUND EYE
# ============================================================

class FlyVision:

    def __init__(self):

        self.width = VISION_WIDTH
        self.height = VISION_HEIGHT

        self.range = VISION_RANGE

        self.fov = math.radians(
            FOV_DEGREES
        )

    # ========================================================
    # ANGLE DIFFERENCE
    # ========================================================

    @staticmethod
    def angle_difference(
        a,
        b
    ):

        d = a - b

        return (
            d + math.pi
        ) % (
            2.0 * math.pi
        ) - math.pi

    # ========================================================
    # PROJECT OBJECT
    # ========================================================

    def project_object(
        self,
        fly,
        object_x,
        object_y,
        channel
    ):

        dx = object_x - fly.x
        dy = object_y - fly.y

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        if distance <= 1e-6:
            return

        if distance > self.range:
            return

        world_angle = math.atan2(
            dy,
            dx
        )

        relative_angle = self.angle_difference(
            world_angle,
            fly.heading
        )

        half_fov = self.fov * 0.5

        if abs(relative_angle) > half_fov:
            return

        # ----------------------------------------------------
        # ANGLE -> RETINA X
        # ----------------------------------------------------

        normalized_angle = (
            relative_angle /
            half_fov
        )

        center_x = (
            0.5 +
            normalized_angle * 0.5
        )

        # ----------------------------------------------------
        # DISTANCE -> APPARENT SIZE
        # ----------------------------------------------------

        size = (
            1.0 /
            (0.15 + distance)
        )

        radius_pixels = int(
            np.clip(
                size * 150.0,
                1.0,
                14.0
            )
        )

        px = int(
            center_x *
            (self.width - 1)
        )

        py = self.height // 2

        # ----------------------------------------------------
        # DRAW GAUSSIAN-LIKE OBJECT
        # ----------------------------------------------------

        y0 = max(
            0,
            py - radius_pixels
        )

        y1 = min(
            self.height,
            py + radius_pixels + 1
        )

        x0 = max(
            0,
            px - radius_pixels
        )

        x1 = min(
            self.width,
            px + radius_pixels + 1
        )

        for y in range(
            y0,
            y1
        ):

            for x in range(
                x0,
                x1
            ):

                dxp = x - px
                dyp = y - py

                r2 = (
                    dxp * dxp +
                    dyp * dyp
                )

                if r2 <= (
                    radius_pixels *
                    radius_pixels
                ):

                    intensity = math.exp(
                        -r2 /
                        max(
                            1.0,
                            radius_pixels *
                            radius_pixels *
                            0.35
                        )
                    )

                    # Slight depth attenuation.
                    depth_factor = max(
                        0.05,
                        1.0 -
                        distance /
                        self.range
                    )

                    value = (
                        intensity *
                        depth_factor
                    )

                    # Stereo separation.
                    if channel == 0:

                        shift = int(
                            normalized_angle *
                            4.0
                        )

                        xx = x + shift

                    else:

                        shift = int(
                            normalized_angle *
                            4.0
                        )

                        xx = x - shift

                    if (
                        0 <= xx <
                        self.width
                    ):

                        # Maximum operation allows
                        # several visual objects.
                        current = self.current_frame[
                            channel,
                            y,
                            xx
                        ]

                        self.current_frame[
                            channel,
                            y,
                            xx
                        ] = max(
                            current,
                            value
                        )

    # ========================================================
    # OBSERVE
    # ========================================================

    def observe(
        self,
        fly,
        sugar_x,
        sugar_y,
        world_width,
        world_height
    ):

        self.current_frame = np.zeros(
            (
                2,
                self.height,
                self.width
            ),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # SUGAR
        # ----------------------------------------------------

        self.project_object(
            fly,
            sugar_x,
            sugar_y,
            channel=0
        )

        self.project_object(
            fly,
            sugar_x,
            sugar_y,
            channel=1
        )

        # ----------------------------------------------------
        # WORLD BOUNDARIES
        #
        # They are visible, but the fly does not receive
        # world coordinates.
        # ----------------------------------------------------

        boundary_objects = [
            (0.0, fly.y),
            (world_width, fly.y),
            (fly.x, 0.0),
            (fly.x, world_height),
        ]

        for bx, by in boundary_objects:

            self.project_object(
                fly,
                bx,
                by,
                channel=0
            )

            self.project_object(
                fly,
                bx,
                by,
                channel=1
            )

        return self.current_frame


# ============================================================
# PROPRIOCEPTION
# ============================================================

class Proprioception:

    def observe(
        self,
        fly
    ):

        speed = math.sqrt(
            fly.vx ** 2 +
            fly.vy ** 2
        )

        return np.array(
            [
                # Velocity
                np.clip(
                    fly.vx / 30.0,
                    -1.0,
                    1.0
                ),

                np.clip(
                    fly.vy / 30.0,
                    -1.0,
                    1.0
                ),

                # Acceleration
                np.clip(
                    fly.ax / 30.0,
                    -1.0,
                    1.0
                ),

                np.clip(
                    fly.ay / 30.0,
                    -1.0,
                    1.0
                ),

                # Angular state
                math.sin(
                    fly.heading
                ),

                math.cos(
                    fly.heading
                ),

                np.clip(
                    fly.angular_velocity / 4.0,
                    -1.0,
                    1.0
                ),

                # Wing state
                np.clip(
                    fly.left_wing,
                    -1.0,
                    1.0
                ),

                np.clip(
                    fly.right_wing,
                    -1.0,
                    1.0
                ),
            ],
            dtype=np.float32
        )