# frozen_string_literal: true

class NeuralNetworkTest < Test::Unit::TestCase
  def test_predict!
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    result = NeuralNetwork
      .new([layer_a, layer_b, layer_c])
      .predict!(inputs)

    assert result < 1.0
  end
end
