import os

import numpy as np
import tensorflow as tf


class CoordinationWeightNetwork(tf.keras.Model):
    """Small two-tower network that produces individual/joint credit scores."""

    def __init__(self, state_dim, context_dim, hidden_dim=4,
                 initial_individual_weight=0.6,
                 initial_joint_weight=0.4, name=None):
        super().__init__(name=name)
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)

        initial_weights = np.asarray(
            [initial_individual_weight, initial_joint_weight],
            dtype=np.float32
        )
        if np.any(initial_weights <= 0.0):
            raise ValueError("Initial coordination weights must be positive.")
        initial_weights /= np.sum(initial_weights)

        self.state_tower = tf.keras.layers.Dense(
            self.hidden_dim, activation="tanh", name=f"{name}_state_tower"
        )
        self.context_tower = tf.keras.layers.Dense(
            self.hidden_dim, activation="tanh", name=f"{name}_context_tower"
        )
        self.score_head = tf.keras.layers.Dense(
            2,
            activation=None,
            kernel_initializer="zeros",
            bias_initializer=tf.keras.initializers.Constant(np.log(initial_weights)),
            name=f"{name}_scores"
        )

        # Build all variables immediately so flatten/restore has a stable layout.
        self(
            tf.zeros((1, self.state_dim), dtype=tf.float32),
            tf.zeros((1, self.context_dim), dtype=tf.float32),
            training=False
        )

    def scores(self, state_embedding, coordination_context, training=False):
        z = tf.convert_to_tensor(state_embedding, dtype=tf.float32)
        e = tf.convert_to_tensor(coordination_context, dtype=tf.float32)
        if len(z.shape) == 1:
            z = tf.expand_dims(z, axis=0)
        if len(e.shape) == 1:
            e = tf.expand_dims(e, axis=0)

        state_features = self.state_tower(z, training=training)
        context_features = self.context_tower(e, training=training)
        interaction = state_features * context_features
        features = tf.concat(
            [state_features, context_features, interaction], axis=-1
        )
        return self.score_head(features, training=training)

    def call(self, state_embedding, coordination_context, training=False):
        return tf.nn.softmax(
            self.scores(state_embedding, coordination_context, training=training),
            axis=-1
        )


class CoordinationWeightGenerator(tf.keras.Model):
    """Owns the truck and drone generators and exposes one CMA-ES vector."""

    TRUCK_CONTEXT_DIM = 6
    DRONE_CONTEXT_DIM = 7

    def __init__(self, state_dim, hidden_dim=4,
                 initial_individual_weight=0.6,
                 initial_joint_weight=0.4):
        super().__init__(name="coordination_weight_generator")
        common = dict(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            initial_individual_weight=initial_individual_weight,
            initial_joint_weight=initial_joint_weight
        )
        self.truck_network = CoordinationWeightNetwork(
            context_dim=self.TRUCK_CONTEXT_DIM,
            name="truck_coordination_weight",
            **common
        )
        self.drone_network = CoordinationWeightNetwork(
            context_dim=self.DRONE_CONTEXT_DIM,
            name="drone_coordination_weight",
            **common
        )

    def for_agent(self, agent_type, state_embedding, coordination_context):
        if agent_type == "truck":
            network = self.truck_network
        elif agent_type == "drone":
            network = self.drone_network
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        # omega is optimized only by CMA-ES. The critic representation is never
        # allowed to receive a gradient through this branch.
        state_embedding = tf.stop_gradient(state_embedding)
        return network(state_embedding, coordination_context, training=False)

    @property
    def parameter_count(self):
        return int(sum(np.prod(v.shape.as_list()) for v in self.trainable_variables))

    def flatten_parameters(self):
        if not self.trainable_variables:
            return np.empty((0,), dtype=np.float64)
        return np.concatenate([
            np.asarray(v.numpy(), dtype=np.float64).reshape(-1)
            for v in self.trainable_variables
        ])

    def restore_parameters(self, flat_parameters):
        flat = np.asarray(flat_parameters, dtype=np.float64).reshape(-1)
        if flat.size != self.parameter_count:
            raise ValueError(
                f"omega has {flat.size} values; expected {self.parameter_count}."
            )

        offset = 0
        for variable in self.trainable_variables:
            size = int(np.prod(variable.shape.as_list()))
            value = flat[offset:offset + size].reshape(variable.shape.as_list())
            variable.assign(value.astype(variable.dtype.as_numpy_dtype))
            offset += size

    def save_parameters(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(path, omega=self.flatten_parameters())

    def load_parameters(self, path):
        with np.load(path) as data:
            self.restore_parameters(data["omega"])
