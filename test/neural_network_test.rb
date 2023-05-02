# frozen_string_literal: true

class NeuralNetworkTest < Test::Unit::TestCase
  def test_learn
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[0, 0, 1], [0, 1, 1], [1, 0, 1]]
    outputs  = Matrix[[0, 1, 1]].transpose.to_a

    network = NeuralNetwork.new([layer_a, layer_b, layer_c])
    network.learn(inputs, outputs, 10_000)

    #assert network.predict!(inputs) > 0.99
  end

  def test_predict!
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    result = NeuralNetwork
      .new([layer_a, layer_b, layer_c])
      .predict!(inputs)

    assert result == 0.9713403945491743
  end
end
