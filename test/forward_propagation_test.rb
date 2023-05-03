# frozen_string_literal: true

class ForwardPropagationTest < Test::Unit::TestCase
  def test_forward
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    outputs = ForwardPropagation.forward([layer_a, layer_b, layer_c], inputs)

    assert_equal(
      Matrix[[0.8807970779778823, 0.8807970779778823, 0.8807970779778823, 0.8807970779778823]],
      outputs[0]
    )

    assert_equal(
      Matrix[[0.9713403945491743, 0.9713403945491743, 0.9713403945491743, 0.9713403945491743]],
      outputs[1]
    )

    assert_equal(
      Matrix[[0.9798730158567087]],
      outputs[2]
    )
  end

  def test_predict!
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    result = ForwardPropagation.predict!([layer_a, layer_b, layer_c], inputs)

    assert result < 1.0
  end
end
