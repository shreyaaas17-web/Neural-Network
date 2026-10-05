from math import exp, log
from random import Random


class Neuron:
    def __init__(self, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def weighted_sum(self, inputs: list[float]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, received {len(inputs)}"
            )
        return sum(weight * value for weight, value in zip(self.weights, inputs)) + self.bias

    @staticmethod
    def activate(value: float) -> float:
        # The two forms are mathematically equivalent; this avoids overflow.
        if value >= 0:
            return 1 / (1 + exp(-value))
        exp_value = exp(value)
        return exp_value / (1 + exp_value)

    @staticmethod
    def activation_derivative(activation: float) -> float:
        return activation * (1 - activation)

    def forward(self, inputs: list[float]) -> tuple[float, float]:
        total = self.weighted_sum(inputs)
        return total, self.activate(total)


class NeuralNetwork:
    def __init__(self, layers: list[list[Neuron]]) -> None:
        if not layers or any(not layer for layer in layers):
            raise ValueError("The network must contain at least one non-empty layer")

        for previous, current in zip(layers, layers[1:]):
            expected_inputs = len(previous)
            if any(len(neuron.weights) != expected_inputs for neuron in current):
                raise ValueError("Each layer must accept one input per previous-layer node")

        self.layers = layers

    def forward(self, inputs: list[float], show_steps: bool = True) -> list[float]:
        activations = inputs
        for layer_index, layer in enumerate(self.layers, start=1):
            next_activations = []
            for node_index, neuron in enumerate(layer, start=1):
                total, activation = neuron.forward(activations)
                next_activations.append(activation)
                if show_steps:
                    print(
                        f"Layer {layer_index}, node {node_index}: "
                        f"weighted sum = {total:.6f}, activation = {activation:.6f}"
                    )
            activations = next_activations
        return activations

    def evaluate(self, samples: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
        if not samples:
            raise ValueError("Evaluation requires at least one sample")

        total_loss = 0.0
        correct = 0
        for inputs, targets in samples:
            outputs = self.forward(inputs, show_steps=False)
            if len(outputs) != len(targets):
                raise ValueError("Each sample must have one target per output node")
            total_loss += self._binary_cross_entropy(outputs, targets)
            correct += all((output >= 0.5) == (target >= 0.5)
                           for output, target in zip(outputs, targets))
        return total_loss / len(samples), correct / len(samples)

    def train(
        self,
        samples: list[tuple[list[float], list[float]]],
        epochs: int,
        learning_rate: float,
        log_every: int = 1000,
    ) -> list[tuple[int, float, float]]:
        if not samples:
            raise ValueError("Training requires at least one sample")
        if epochs < 1 or learning_rate <= 0 or log_every < 1:
            raise ValueError("epochs, learning_rate, and log_every must be positive")

        input_count = len(self.layers[0][0].weights)
        output_count = len(self.layers[-1])
        if any(len(inputs) != input_count or len(targets) != output_count
               for inputs, targets in samples):
            raise ValueError("Sample input and target sizes must match the network")

        history = []
        for epoch in range(1, epochs + 1):
            for inputs, targets in samples:
                layer_activations = [inputs]
                for layer in self.layers:
                    layer_activations.append([
                        neuron.activate(neuron.weighted_sum(layer_activations[-1]))
                        for neuron in layer
                    ])

                deltas: list[list[float]] = [[] for _ in self.layers]
                deltas[-1] = [
                    output - target
                    for output, target in zip(layer_activations[-1], targets)
                ]
                for layer_index in range(len(self.layers) - 2, -1, -1):
                    deltas[layer_index] = [
                        sum(
                            next_neuron.weights[node_index] * next_delta
                            for next_neuron, next_delta in zip(
                                self.layers[layer_index + 1], deltas[layer_index + 1]
                            )
                        ) * Neuron.activation_derivative(activation)
                        for node_index, activation in enumerate(layer_activations[layer_index + 1])
                    ]

                for layer_index, layer in enumerate(self.layers):
                    inputs_to_layer = layer_activations[layer_index]
                    for neuron, delta in zip(layer, deltas[layer_index]):
                        neuron.weights = [
                            weight - learning_rate * delta * input_value
                            for weight, input_value in zip(neuron.weights, inputs_to_layer)
                        ]
                        neuron.bias -= learning_rate * delta

            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                loss, accuracy = self.evaluate(samples)
                history.append((epoch, loss, accuracy))
                print(
                    f"Epoch {epoch:5d}/{epochs}: "
                    f"loss = {loss:.6f}, accuracy = {accuracy:.0%}"
                )
        return history

    @staticmethod
    def _binary_cross_entropy(outputs: list[float], targets: list[float]) -> float:
        epsilon = 1e-12
        return -sum(
            target * log(max(output, epsilon))
            + (1 - target) * log(max(1 - output, epsilon))
            for output, target in zip(outputs, targets)
        ) / len(outputs)


def make_xor_network(seed: int = 3) -> NeuralNetwork:
    generator = Random(seed)
    layers = []
    for input_count, node_count in [(2, 4), (4, 1)]:
        layers.append([
            Neuron(
                [generator.uniform(-1, 1) for _ in range(input_count)],
                generator.uniform(-1, 1),
            )
            for _ in range(node_count)
        ])
    return NeuralNetwork(layers)


def main() -> None:
    xor_samples = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    network = make_xor_network()

    initial_loss, initial_accuracy = network.evaluate(xor_samples)
    initial_weights = network.layers[0][0].weights.copy()
    initial_bias = network.layers[0][0].bias
    print("XOR training: 2 inputs -> 4 sigmoid hidden nodes -> 1 sigmoid output")
    print("Sigmoid activation: 1 / (1 + exp(-z)); derivative: a * (1 - a)")
    print(f"Before training: loss = {initial_loss:.6f}, accuracy = {initial_accuracy:.0%}")
    network.train(xor_samples, epochs=5000, learning_rate=1.0, log_every=500)
    print(
        f"First hidden node weights: {initial_weights} -> "
        f"{network.layers[0][0].weights}"
    )
    print(
        f"First hidden node bias: {initial_bias:.6f} -> "
        f"{network.layers[0][0].bias:.6f}"
    )

    final_loss, final_accuracy = network.evaluate(xor_samples)
    print("\nPredictions after training:")
    for inputs, target in xor_samples:
        output = network.forward(inputs, show_steps=False)[0]
        prediction = int(output >= 0.5)
        print(f"{inputs} -> {output:.6f} (class {prediction}, target {int(target[0])})")
    print(f"Final performance: loss = {final_loss:.6f}, accuracy = {final_accuracy:.0%}")


if __name__ == "__main__":
    main()