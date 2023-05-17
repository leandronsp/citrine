# frozen_string_literal: true

class ForwardPropagationTest < Test::Unit::TestCase
  def test_forward
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    layers = ForwardPropagation.call([layer_a, layer_b, layer_c], inputs)

    assert_equal(
      Matrix[[0.8807970779778823, 0.8807970779778823, 0.8807970779778823, 0.8807970779778823]],
      layers[0].result
    )

    assert_equal(
      Matrix[[0.9713403945491743, 0.9713403945491743, 0.9713403945491743, 0.9713403945491743]],
      layers[1].result
    )

    assert_equal(
      Matrix[[0.9798730158567087]],
      layers[2].result
    )
  end
end
