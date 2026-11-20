import os
import random
import time

import numpy as np
import torch


from environment.fly_env import FlyEnvironment

from brain.model import FlyBrain

from rl.ppo import (
    PPO,
    RolloutBuffer
)

from config import (
    SEED,
    CHECKPOINT_DIR,
    LOG_DIR,
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


# ============================================================
# NEURAL STATE FILE
# ============================================================

NEURAL_STATE_PATH = os.path.join(
    LOG_DIR,
    "brain_state.npz"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("SINEK FROM SCRATCH")
print("=" * 70)

print(
    f"Device: {device}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    print(
        "GPU kullanilmiyor."
    )

print("=" * 70)


# ============================================================
# ENVIRONMENT
# ============================================================

env = FlyEnvironment()


# ============================================================
# BRAIN
# ============================================================

brain = FlyBrain(
    proprioception_size=9,
    action_size=9,
    snn_size=128,
).to(device)


# ============================================================
# PPO
# ============================================================

agent = PPO(

    brain=brain,

    lr=3e-4,

    gamma=0.99,

    gae_lambda=0.95,

    clip_eps=0.2,

    epochs=4,

    batch_size=128,

    value_coef=0.5,

    entropy_coef=0.01,
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

ROLLOUT_STEPS = 2048

UPDATES = 500


# ============================================================
# LIVE VISUALIZATION
# ============================================================

# Her kaç environment step'te
# beynin görüntüsünü güncelleyelim?

VISUALIZE_EVERY = 100


# ============================================================
# BUFFER
# ============================================================

buffer = RolloutBuffer()


# ============================================================
# INITIAL RESET
# ============================================================

observation = env.reset()


episode_reward = 0.0

episode_length = 0

episode = 0

global_step = 0


best_distance = float(
    "inf"
)


best_reward = -float(
    "inf"
)


# ============================================================
# HELPER
# ============================================================

def save_brain_state(
    brain_output,
    brain,
    episode,
    global_step,
    episode_reward,
):
    """
    Training sırasında SNN durumunu
    brain_view.py'nin okuyabileceği
    şekilde kaydeder.

    Windows dosya kilitlenmesini
    önlemek için:

    1. benzersiz temp dosyası
    2. temp dosyaya yaz
    3. temp dosyayı kapat
    4. os.replace
    """

    try:

        spike_rate = (
            brain_output["spike_rate"]
            .detach()
            .squeeze(0)
            .cpu()
            .numpy()
            .copy()
        )

        membrane = (
            brain_output["membrane"]
            .detach()
            .squeeze(0)
            .cpu()
            .numpy()
            .copy()
        )

        output_prob = torch.softmax(
            brain_output["logits"],
            dim=-1
        )

        output_prob = (
            output_prob
            .detach()
            .squeeze(0)
            .cpu()
            .numpy()
            .copy()
        )

        # ----------------------------------------------------
        # İlk actor katmanının ağırlıkları
        # ----------------------------------------------------

        output_weights = (
            brain.actor[0]
            .weight
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        # ----------------------------------------------------
        # Benzersiz temporary dosya
        # ----------------------------------------------------

        temp_path = (
            NEURAL_STATE_PATH
            +
            f".tmp_{os.getpid()}"
        )

        # ----------------------------------------------------
        # WRITE
        # ----------------------------------------------------

        with open(
            temp_path,
            "wb"
        ) as f:

            np.savez(
                f,

                spike_rate=spike_rate,

                membrane=membrane,

                output_prob=output_prob,

                output_weights=output_weights,

                episode=np.array(
                    episode,
                    dtype=np.int64
                ),

                step=np.array(
                    global_step,
                    dtype=np.int64
                ),

                episode_reward=np.array(
                    episode_reward,
                    dtype=np.float32
                ),
            )

            # Verinin gerçekten
            # dosyaya yazılmasını zorla.

            f.flush()

            os.fsync(
                f.fileno()
            )

        # ----------------------------------------------------
        # WINDOWS SAFE REPLACE
        # ----------------------------------------------------

        for attempt in range(10):

            try:

                os.replace(
                    temp_path,
                    NEURAL_STATE_PATH
                )

                break

            except PermissionError:

                if attempt == 9:

                    # Viewer dosyayı kilitlediyse
                    # eğitimi durdurma.

                    try:

                        if os.path.exists(
                            temp_path
                        ):

                            os.remove(
                                temp_path
                            )

                    except OSError:

                        pass

                    return

                time.sleep(
                    0.01
                )

    except Exception as e:

        # Visualization yüzünden
        # eğitim asla durmasın.

        print(
            f"[VISUALIZATION WARNING] "
            f"{e}"
        )


# ============================================================
# TRAINING LOOP
# ============================================================

for update in range(
    1,
    UPDATES + 1
):

    buffer.clear()

    update_reward = 0.0


    # ========================================================
    # EXPERIENCE COLLECTION
    # ========================================================

    for step in range(
        ROLLOUT_STEPS
    ):

        global_step += 1


        # ----------------------------------------------------
        # OBSERVATION
        # ----------------------------------------------------

        vision_np = observation[
            "vision"
        ]

        proprio_np = observation[
            "proprioception"
        ]


        vision = torch.tensor(
            vision_np,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)


        proprio = torch.tensor(
            proprio_np,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)


        # ----------------------------------------------------
        # BRAIN
        # ----------------------------------------------------

        with torch.no_grad():

            (
                action,
                log_probability,
                value,
                brain_output
            ) = brain.get_action(
                vision,
                proprio
            )


        action_int = int(
            action.item()
        )


        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(
            action_int
        )


        done = (
            terminated
            or
            truncated
        )


        # ----------------------------------------------------
        # STORE EXPERIENCE
        # ----------------------------------------------------

        buffer.vision.append(
            vision_np
        )

        buffer.proprio.append(
            proprio_np
        )

        buffer.actions.append(
            action_int
        )

        buffer.log_probs.append(
            float(
                log_probability.item()
            )
        )

        buffer.rewards.append(
            float(reward)
        )

        buffer.dones.append(
            float(done)
        )

        buffer.values.append(
            float(
                value.item()
            )
        )


        # ----------------------------------------------------
        # REWARD
        # ----------------------------------------------------

        episode_reward += float(
            reward
        )

        update_reward += float(
            reward
        )

        episode_length += 1


        # ----------------------------------------------------
        # LIVE BRAIN
        # ----------------------------------------------------

        if (
            global_step
            %
            VISUALIZE_EVERY
            == 0
        ):

            save_brain_state(
                brain_output,
                brain,
                episode,
                global_step,
                episode_reward
            )


        # ----------------------------------------------------
        # NEXT OBSERVATION
        # ----------------------------------------------------

        observation = (
            next_observation
        )


        # ====================================================
        # EPISODE END
        # ====================================================

        if done:

            episode += 1


            distance = float(
                info.get(
                    "distance",
                    999999.0
                )
            )


            # ------------------------------------------------
            # BEST DISTANCE
            # ------------------------------------------------

            if (
                distance
                <
                best_distance
            ):

                best_distance = (
                    distance
                )


            # ------------------------------------------------
            # BEST REWARD
            # ------------------------------------------------

            if (
                episode_reward
                >
                best_reward
            ):

                best_reward = (
                    episode_reward
                )


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            success = bool(
                info.get(
                    "success",
                    False
                )
            )


            success_text = (
                "SUCCESS"
                if success
                else ""
            )


            # ------------------------------------------------
            # PRINT
            # ------------------------------------------------

            print(
                f"Episode "
                f"{episode:05d} | "

                f"Reward "
                f"{episode_reward:9.3f} | "

                f"Distance "
                f"{distance:7.2f} | "

                f"Steps "
                f"{episode_length:4d} | "

                f"Best "
                f"{best_distance:7.2f} "

                f"{success_text}"
            )


            # ------------------------------------------------
            # RESET EPISODE
            # ------------------------------------------------

            episode_reward = 0.0

            episode_length = 0

            observation = (
                env.reset()
            )


    # ========================================================
    # PPO UPDATE
    # ========================================================

    last_vision = observation[
        "vision"
    ]

    last_proprio = observation[
        "proprioception"
    ]


    loss = agent.update(

        buffer,

        last_vision,

        last_proprio,

        False
    )


    # ========================================================
    # FINAL VISUALIZATION UPDATE
    # ========================================================

    with torch.no_grad():

        debug_vision = torch.tensor(
            observation["vision"],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)


        debug_proprio = torch.tensor(
            observation["proprioception"],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)


        debug_output = brain(
            debug_vision,
            debug_proprio
        )


    save_brain_state(
        debug_output,
        brain,
        episode,
        global_step,
        episode_reward
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if update % 10 == 0:

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"fly_brain_update_"
            f"{update:05d}.pt"
        )


        torch.save(
            {
                "update":
                    update,

                "episode":
                    episode,

                "global_step":
                    global_step,

                "model_state_dict":
                    brain.state_dict(),

                "optimizer_state_dict":
                    agent.optimizer.state_dict(),

                "best_distance":
                    best_distance,

                "best_reward":
                    best_reward,
            },

            checkpoint_path
        )


        print()
        print(
            f"[CHECKPOINT] "
            f"{checkpoint_path}"
        )

        print(
            f"[UPDATE] "
            f"{update}/{UPDATES} "
            f"| Loss: {loss:.4f}"
        )

        print()