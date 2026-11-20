import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    SNN_SIZE,
    SNN_BETA,
    SNN_THRESHOLD,
    SNN_RESET,
    SNN_SYNAPTIC_DECAY,
    SNN_TIME_STEPS,
    CNN_FEATURE_SIZE,
    PROPRIO_FEATURE_SIZE,
    ACTION_SIZE,
)


# ============================================================
# CNN VISUAL SYSTEM
# ============================================================

class VisualCNN(
    nn.Module
):

    def __init__(
        self,
        in_channels=2
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels,
                32,
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm2d(32),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.GELU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.GELU(),

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                (4, 4)
            ),
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                CNN_FEATURE_SIZE
            ),

            nn.LayerNorm(
                CNN_FEATURE_SIZE
            ),

            nn.GELU()
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        return self.fc(
            x
        )


# ============================================================
# PROPRIOCEPTIVE SYSTEM
# ============================================================

class ProprioEncoder(
    nn.Module
):

    def __init__(
        self,
        input_size=9
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_size,
                64
            ),

            nn.LayerNorm(64),

            nn.GELU(),

            nn.Linear(
                64,
                PROPRIO_FEATURE_SIZE
            ),

            nn.LayerNorm(
                PROPRIO_FEATURE_SIZE
            ),

            nn.GELU()
        )

    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# SNN STATE
# ============================================================

class SNNState:

    def __init__(
        self,
        membrane,
        synaptic,
        spikes
    ):

        self.membrane = membrane
        self.synaptic = synaptic
        self.spikes = spikes

    # --------------------------------------------------------
    # DETACH
    # --------------------------------------------------------

    def detach(self):

        return SNNState(
            self.membrane.detach(),
            self.synaptic.detach(),
            self.spikes.detach()
        )

    # --------------------------------------------------------
    # CLONE
    # --------------------------------------------------------

    def clone(self):

        return SNNState(
            self.membrane.clone(),
            self.synaptic.clone(),
            self.spikes.clone()
        )


# ============================================================
# RECURRENT LIF SNN
# ============================================================

class RecurrentLIF(
    nn.Module
):

    def __init__(
        self,
        input_size,
        hidden_size=SNN_SIZE
    ):

        super().__init__()

        self.input_layer = nn.Linear(
            input_size,
            hidden_size
        )

        self.recurrent = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False
        )

        self.beta = SNN_BETA

        self.threshold = SNN_THRESHOLD

        self.reset_value = SNN_RESET

        self.synaptic_decay = (
            SNN_SYNAPTIC_DECAY
        )

        self.hidden_size = hidden_size

    # ========================================================
    # INITIAL STATE
    # ========================================================

    def initial_state(
        self,
        batch_size,
        device,
        dtype=torch.float32
    ):

        zeros = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device,
            dtype=dtype
        )

        return SNNState(
            membrane=zeros.clone(),
            synaptic=zeros.clone(),
            spikes=zeros.clone()
        )

    # ========================================================
    # SINGLE STEP
    # ========================================================

    def step(
        self,
        x,
        state
    ):

        input_current = self.input_layer(
            x
        )

        recurrent_current = self.recurrent(
            state.spikes
        )

        state.synaptic = (
            self.synaptic_decay *
            state.synaptic
            +
            input_current
            +
            recurrent_current
        )

        state.membrane = (
            self.beta *
            state.membrane
            +
            state.synaptic
        )

        spikes = (
            state.membrane >=
            self.threshold
        ).float()

        state.membrane = torch.where(
            spikes > 0,
            torch.full_like(
                state.membrane,
                self.reset_value
            ),
            state.membrane
        )

        state.spikes = spikes

        return (
            spikes,
            state
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x,
        state=None,
        time_steps=SNN_TIME_STEPS
    ):

        batch_size = x.shape[0]

        if state is None:

            state = self.initial_state(
                batch_size,
                x.device,
                x.dtype
            )

        spike_history = []

        membrane_history = []

        synaptic_history = []

        for _ in range(
            time_steps
        ):

            (
                spikes,
                state
            ) = self.step(
                x,
                state
            )

            spike_history.append(
                spikes
            )

            membrane_history.append(
                state.membrane.clone()
            )

            synaptic_history.append(
                state.synaptic.clone()
            )

        spike_history = torch.stack(
            spike_history,
            dim=1
        )

        membrane_history = torch.stack(
            membrane_history,
            dim=1
        )

        synaptic_history = torch.stack(
            synaptic_history,
            dim=1
        )

        spike_rate = (
            spike_history.mean(
                dim=1
            )
        )

        return {
            "spike_rate":
                spike_rate,

            "membrane":
                state.membrane,

            "synaptic":
                state.synaptic,

            "spikes":
                state.spikes,

            "spike_history":
                spike_history,

            "membrane_history":
                membrane_history,

            "synaptic_history":
                synaptic_history,

            "state":
                state
        }


# ============================================================
# COMPLETE FLY BRAIN
# ============================================================

class FlyBrain(
    nn.Module
):

    def __init__(
        self,
        proprioception_size=9,
        action_size=ACTION_SIZE,
        snn_size=SNN_SIZE
    ):

        super().__init__()

        self.cnn = VisualCNN(
            in_channels=2
        )

        self.proprio = ProprioEncoder(
            proprioception_size
        )

        fusion_size = (
            CNN_FEATURE_SIZE +
            PROPRIO_FEATURE_SIZE
        )

        self.snn = RecurrentLIF(
            input_size=fusion_size,
            hidden_size=snn_size
        )

        self.actor = nn.Sequential(

            nn.Linear(
                snn_size,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                action_size
            )
        )

        self.critic = nn.Sequential(

            nn.Linear(
                snn_size,
                128
            ),

            nn.GELU(),

            nn.Linear(
                128,
                1
            )
        )

    # ========================================================
    # INITIAL STATE
    # ========================================================

    def initial_state(
        self,
        batch_size,
        device=None
    ):

        if device is None:

            device = next(
                self.parameters()
            ).device

        return self.snn.initial_state(
            batch_size,
            device
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        vision,
        proprioception,
        state=None
    ):

        visual_features = self.cnn(
            vision
        )

        proprio_features = self.proprio(
            proprioception
        )

        combined = torch.cat(
            [
                visual_features,
                proprio_features
            ],
            dim=1
        )

        snn_output = self.snn(
            combined,
            state
        )

        neural_state = (
            snn_output["state"]
        )

        logits = self.actor(
            snn_output["spike_rate"]
        )

        value = self.critic(
            snn_output["spike_rate"]
        )

        return {

            "logits":
                logits,

            "value":
                value,

            "spike_rate":
                snn_output[
                    "spike_rate"
                ],

            "membrane":
                snn_output[
                    "membrane"
                ],

            "synaptic":
                snn_output[
                    "synaptic"
                ],

            "spikes":
                snn_output[
                    "spikes"
                ],

            "spike_history":
                snn_output[
                    "spike_history"
                ],

            "membrane_history":
                snn_output[
                    "membrane_history"
                ],

            "synaptic_history":
                snn_output[
                    "synaptic_history"
                ],

            "state":
                neural_state
        }

    # ========================================================
    # ACTION
    # ========================================================

    def get_action(
        self,
        vision,
        proprioception,
        state=None
    ):

        output = self.forward(
            vision,
            proprioception,
            state
        )

        distribution = torch.distributions.Categorical(
            logits=output["logits"]
        )

        action = distribution.sample()

        log_probability = (
            distribution.log_prob(
                action
            )
        )

        return (
            action,
            log_probability,
            output["value"],
            output
        )