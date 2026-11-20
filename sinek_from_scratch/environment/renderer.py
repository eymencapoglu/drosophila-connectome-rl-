# ============================================================
# LIVE ENVIRONMENT RENDERER
# ============================================================

import pygame
import numpy as np

from config import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_SCALE,
    FPS,
)


class FlyRenderer:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            )
        )

        pygame.display.set_caption(
            "Sinek From Scratch - 2D Environment"
        )

        self.clock = pygame.time.Clock()

        self.running = True

        self.font = pygame.font.SysFont(
            "Arial",
            20
        )

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def render(
        self,
        environment,
        action=0,
        reward=0.0,
        episode=0,
    ):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                self.running = False

        self.screen.fill(
            (0, 0, 0)
        )

        # ----------------------------------------------------
        # OBSERVATION
        # ----------------------------------------------------

        observation = environment.get_observation()

        image = observation[0]

        # ----------------------------------------------------
        # DRAW PIXELS
        # ----------------------------------------------------

        for y in range(
            image.shape[0]
        ):

            for x in range(
                image.shape[1]
            ):

                value = image[y, x]

                if value <= 0:
                    continue

                intensity = int(
                    value * 255
                )

                color = (
                    intensity,
                    intensity,
                    intensity
                )

                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        x * WINDOW_SCALE,
                        y * WINDOW_SCALE,
                        WINDOW_SCALE,
                        WINDOW_SCALE
                    )
                )

        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        info = environment.fly.get_state()

        lines = [
            f"Episode: {episode}",
            f"Step: {environment.step_count}",
            f"Action: {action}",
            f"Reward: {reward:.4f}",
            f"X: {info['x']:.2f}",
            f"Y: {info['y']:.2f}",
            f"VX: {info['vx']:.2f}",
            f"VY: {info['vy']:.2f}",
            f"Energy: {info['energy']:.3f}",
        ]

        # Bilgi panelini sağ tarafa
        # ekranın altına yerleştiriyoruz.

        y_position = 10

        for line in lines:

            text = self.font.render(
                line,
                True,
                (255, 255, 255)
            )

            self.screen.blit(
                text,
                (
                    10,
                    y_position
                )
            )

            y_position += 22

        pygame.display.flip()

        self.clock.tick(FPS)

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(self):

        pygame.quit()