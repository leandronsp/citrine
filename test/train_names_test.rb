# frozen_string_literal: true

class TrainNamesTest < Test::Unit::TestCase
  def test_train_names
    neuron_a = Neuron.new([-0.16595599, -0.70648822, -0.20646505])
    neuron_b = Neuron.new([0.44064899, -0.81532281, 0.07763347])
    neuron_c = Neuron.new([-0.99977125, -0.62747958, -0.16161097])
    neuron_d = Neuron.new([-0.39533485, -0.30887855, 0.370439])
    layer_a = Layer.new([neuron_a, neuron_b, neuron_c, neuron_d])

    neuron_e = Neuron.new([-0.16595599, -0.70648822, -0.20646505, -0.34093502])
    neuron_f = Neuron.new([0.44064899, -0.81532281, 0.07763347, 0.44093502])
    neuron_g = Neuron.new([-0.99977125, -0.62747958, -0.16161097, 0.14093502])
    neuron_h = Neuron.new([-0.39533485, -0.30887855, 0.370439, -0.54093502])
    layer_b = Layer.new([neuron_e, neuron_f, neuron_g, neuron_h])

    neuron_i = Neuron.new([-0.5910955, 0.75623487, -0.94522481, 0.64093502])
    layer_c = Layer.new([neuron_i])

    names = [
      ['Leandro', [1, 1, 0]],
      ['Cassia', [2, 2, 1]],
      ['Joao', [3, 3, 0]],
      ['Maria', [4, 4, 1]],
      ['Gustavo', [5, 5, 0]],
      ['Ana', [6, 6, 1]],
      ['Sabrina', [7, 7, 1]]
    ]

    inputs = names.to_h.values
    targets  = Matrix[[0, 1, 0, 1, 0, 1, 1]].transpose.to_a

    network = NeuralNetwork.new([layer_a, layer_b, layer_c])
    network.learn(inputs, targets, 2_000)

    %w[Leandro Joao Gustavo]
      .map { |name| network.predict!([names.to_h[name]]) }
      .then { |results| assert results.max < 0.02 }

    %w[Cassia Maria Ana Sabrina]
      .map { |name| network.predict!([names.to_h[name]]) }
      .then { |results| assert results.max > 0.98 }
  end
end
