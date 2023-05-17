# frozen_string_literal: true

class TrainXORTest < Test::Unit::TestCase
  def test_train_xor_gate
    # First Layer (4 neurons -> 3 weights)
    neuron_a = Neuron.new([-0.16595599, -0.70648822, -0.20646505])
    neuron_b = Neuron.new([0.44064899, -0.81532281, 0.07763347])
    neuron_c = Neuron.new([-0.99977125, -0.62747958, -0.16161097])
    neuron_d = Neuron.new([-0.39533485, -0.30887855, 0.370439])
    layer_a = Layer.new([neuron_a, neuron_b, neuron_c, neuron_d])

    # Middle Layer (4 neurons -> 4 weights)
    neuron_e = Neuron.new([-0.16595599, -0.70648822, -0.20646505, -0.34093502])
    neuron_f = Neuron.new([0.44064899, -0.81532281, 0.07763347, 0.44093502])
    neuron_g = Neuron.new([-0.99977125, -0.62747958, -0.16161097, 0.14093502])
    neuron_h = Neuron.new([-0.39533485, -0.30887855, 0.370439, -0.54093502])
    layer_b = Layer.new([neuron_e, neuron_f, neuron_g, neuron_h])

    # Last Layer (1 neuron -> 4 weights)
    neuron_i = Neuron.new([-0.5910955, 0.75623487, -0.94522481, 0.64093502])
    layer_c = Layer.new([neuron_i])

    inputs  = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    targets = Matrix[[0, 1, 1, 1, 1, 0, 0]].transpose.to_a

    network = NeuralNetwork.new([layer_a, layer_b, layer_c])
    network.learn(inputs, targets, 2_000)

    assert_equal 0.05, network.predict([[1, 1, 0]]).round(2)
  end
end
