import os
import time
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

STATE_PATH = (
    r"C:\Users\Eymen\Desktop\SİNEK"
    r"\sinek_from_scratch"
    r"\logs"
    r"\brain_state.npz"
)


# ============================================================
# NETWORK SIZE
# ============================================================

SNN_NEURONS = 128
ACTION_NEURONS = 9


# ============================================================
# 3D BRAIN GEOMETRY
# ============================================================

rng = np.random.default_rng(42)

angles = rng.uniform(
    0,
    2 * np.pi,
    SNN_NEURONS
)

radius = rng.uniform(
    1.0,
    4.0,
    SNN_NEURONS
)

z = rng.uniform(
    -4.0,
    4.0,
    SNN_NEURONS
)

x = np.cos(angles) * radius
y = np.sin(angles) * radius

snn_positions = np.column_stack(
    [
        x,
        y,
        z
    ]
)


# ============================================================
# ACTION / MOTOR NEURONS
# ============================================================

action_positions = np.column_stack(
    [
        np.ones(ACTION_NEURONS) * 7.0,

        np.linspace(
            -4.0,
            4.0,
            ACTION_NEURONS
        ),

        np.zeros(
            ACTION_NEURONS
        )
    ]
)


ACTION_NAMES = [
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "LEFT WING",
    "RIGHT WING",
    "BRAKE",
]


# ============================================================
# MATPLOTLIB
# ============================================================

plt.ion()

fig = plt.figure(
    figsize=(13, 9)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# LAST FILE MODIFICATION
# ============================================================

last_modified = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # FILE DOES NOT EXIST
    # --------------------------------------------------------

    if not os.path.exists(
        STATE_PATH
    ):

        ax.clear()

        ax.set_title(
            "SİNEK SNN — TRAINING BEKLENİYOR"
        )

        ax.set_xlabel(
            "Neural X"
        )

        ax.set_ylabel(
            "Neural Y"
        )

        ax.set_zlabel(
            "Neural Z"
        )

        plt.pause(
            0.1
        )

        continue


    # --------------------------------------------------------
    # CHECK FILE TIME
    # --------------------------------------------------------

    try:

        modified = os.path.getmtime(
            STATE_PATH
        )

    except OSError:

        time.sleep(
            0.05
        )

        continue


    # --------------------------------------------------------
    # SAME FRAME
    # --------------------------------------------------------

    if modified <= last_modified:

        plt.pause(
            0.03
        )

        continue


    # --------------------------------------------------------
    # LOAD FILE SAFELY
    # IMPORTANT:
    # with np.load(...) closes the file immediately.
    # This prevents Windows file locking.
    # --------------------------------------------------------

    try:

        with np.load(
            STATE_PATH,
            allow_pickle=False
        ) as data:

            spike_rate = np.array(
                data["spike_rate"],
                copy=True
            )

            membrane = np.array(
                data["membrane"],
                copy=True
            )

            output_prob = np.array(
                data["output_prob"],
                copy=True
            )

            output_weights = None

            if "output_weights" in data:

                output_weights = np.array(
                    data["output_weights"],
                    copy=True
                )

            step_number = 0

            if "step" in data:

                step_number = int(
                    data["step"]
                )

            episode_number = 0

            if "episode" in data:

                episode_number = int(
                    data["episode"]
                )

            reward = 0.0

            if "episode_reward" in data:

                reward = float(
                    data["episode_reward"]
                )

    except (
        OSError,
        ValueError,
        EOFError,
        KeyError
    ):

        # Training may be writing the file.
        # Simply wait for the next frame.

        time.sleep(
            0.03
        )

        continue


    last_modified = modified


    # ========================================================
    # NORMALIZE ACTIVITY
    # ========================================================

    spike_rate = np.nan_to_num(
        spike_rate,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    membrane = np.nan_to_num(
        membrane,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    output_prob = np.nan_to_num(
        output_prob,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )


    # ========================================================
    # DRAW
    # ========================================================

    ax.clear()


    # ========================================================
    # SNN NEURONS
    # ========================================================

    ax.scatter(
        snn_positions[:, 0],
        snn_positions[:, 1],
        snn_positions[:, 2],

        s=20 + spike_rate * 500,

        c=spike_rate,

        cmap="plasma",

        vmin=0.0,
        vmax=1.0,

        alpha=0.90
    )


    # ========================================================
    # ACTION NEURONS
    # ========================================================

    max_probability = max(
        float(output_prob.max()),
        1e-6
    )

    ax.scatter(
        action_positions[:, 0],
        action_positions[:, 1],
        action_positions[:, 2],

        s=80 + (
            output_prob /
            max_probability
        ) * 400,

        c=output_prob,

        cmap="viridis",

        vmin=0.0,
        vmax=max_probability,

        edgecolors="black",
        linewidths=0.8
    )


    # ========================================================
    # SYNAPTIC CONNECTIONS
    # ========================================================

    if output_weights is not None:

        # Eğer actor ilk layer 64x128 ise
        # her action için doğrudan çizim yapamayız.
        #
        # Bu nedenle şimdilik en güçlü bağlantıları
        # görselleştiriyoruz.

        weight_matrix = np.asarray(
            output_weights
        )

        if (
            weight_matrix.ndim == 2
            and
            weight_matrix.shape[1]
            == SNN_NEURONS
        ):

            for output_index in range(
                min(
                    ACTION_NEURONS,
                    weight_matrix.shape[0]
                )
            ):

                row = weight_matrix[
                    output_index
                ]

                top_indices = np.argsort(
                    np.abs(row)
                )[-5:]

                for neuron_index in top_indices:

                    weight = row[
                        neuron_index
                    ]

                    alpha = min(
                        0.65,
                        0.10 +
                        abs(weight)
                    )

                    ax.plot(
                        [
                            snn_positions[
                                neuron_index,
                                0
                            ],

                            action_positions[
                                output_index,
                                0
                            ]
                        ],

                        [
                            snn_positions[
                                neuron_index,
                                1
                            ],

                            action_positions[
                                output_index,
                                1
                            ]
                        ],

                        [
                            snn_positions[
                                neuron_index,
                                2
                            ],

                            action_positions[
                                output_index,
                                2
                            ]
                        ],

                        alpha=alpha,

                        linewidth=0.7
                    )


    # ========================================================
    # MOST ACTIVE NEURONS
    # ========================================================

    top_neurons = np.argsort(
        spike_rate
    )[-10:]


    for neuron_index in top_neurons:

        if spike_rate[
            neuron_index
        ] > 0:

            ax.text(
                snn_positions[
                    neuron_index,
                    0
                ],

                snn_positions[
                    neuron_index,
                    1
                ],

                snn_positions[
                    neuron_index,
                    2
                ],

                str(neuron_index),

                fontsize=7
            )


    # ========================================================
    # MOST PROBABLE ACTION
    # ========================================================

    selected_action = int(
        np.argmax(
            output_prob
        )
    )


    selected_probability = float(
        output_prob[
            selected_action
        ]
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        "SİNEK — LIVE LIF-SNN ACTIVITY\n"
        f"Episode: {episode_number}   "
        f"Step: {step_number}   "
        f"Reward: {reward:.3f}\n"
        f"Action: "
        f"{ACTION_NAMES[selected_action]}   "
        f"P={selected_probability:.3f}"
    )


    # ========================================================
    # AXIS
    # ========================================================

    ax.set_xlabel(
        "Neural X"
    )

    ax.set_ylabel(
        "Neural Y"
    )

    ax.set_zlabel(
        "Neural Z"
    )


    ax.set_xlim(
        -6,
        8
    )

    ax.set_ylim(
        -6,
        6
    )

    ax.set_zlim(
        -6,
        6
    )


    # ========================================================
    # CAMERA
    # ========================================================

    ax.view_init(
        elev=25,
        azim=45
    )


    # ========================================================
    # DRAW
    # ========================================================

    plt.pause(
        0.001
    )