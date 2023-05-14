# frozen_string_literal: true

class NeuralNetworkTest < Test::Unit::TestCase
  def setup
    @neuron_a = Neuron.new([-0.16595599, -0.70648822, -0.20646505])
    @neuron_b = Neuron.new([0.44064899, -0.81532281, 0.07763347])
    @neuron_c = Neuron.new([-0.99977125, -0.62747958, -0.16161097])
    @neuron_d = Neuron.new([-0.39533485, -0.30887855, 0.370439])
    @neuron_e = Neuron.new([-0.5910955, 0.75623487, -0.94522481, 0.34093502])

    @layer_a = Layer.new([@neuron_a, @neuron_b, @neuron_c, @neuron_d])
    @layer_b = Layer.new([@neuron_e])

    # XOR gate
    @inputs = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    @targets  = Matrix[[0, 1, 1, 1, 1, 0, 0]].transpose.to_a
  end

  def test_learn_100_times
    network = NeuralNetwork.new([@layer_a, @layer_b])
    network.learn(@inputs, @targets, 100)

    assert_equal(
      0.81,
      network.predict!([[1, 1, 0]]).round(2)
    )
  end

  #def test_learn_1000_times
  #  network = NeuralNetwork.new([@layer_a, @layer_b])
  #  network.learn(@inputs, @targets, 1_000)

  #  assert_equal(
  #    0.51,
  #    network.predict!([[1, 1, 0]]).round(2)
  #  )
  #end

  #def test_learn_2000_times
  #  network = NeuralNetwork.new([@layer_a, @layer_b])
  #  network.learn(@inputs, @targets, 2_000)

  #  assert_equal(
  #    0.07,
  #    network.predict!([[1, 1, 0]]).round(2)
  #  )
  #end

  #def test_learn_multiple_times_in_sequence
  #  network = NeuralNetwork.new([@layer_a, @layer_b])
  #  network.learn(@inputs, @targets, 2_000)
  #  network.learn(@inputs, @targets, 2_000)
  #  network.learn(@inputs, @targets, 2_000)

  #  assert_equal(
  #    0.03,
  #    network.predict!([[1, 1, 0]]).round(2)
  #  )
  #end
end
