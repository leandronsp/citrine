# frozen_string_literal: true

class LayerTest < Test::Unit::TestCase
  def test_convert_layer_to_matrix_4_3
    layer = Layer.build(
      number_of_neurons: 4,
      number_of_weights: 3
    )

    assert_equal(
      [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
      layer.to_matrix
    )
  end

  def test_convert_layer_to_matrix_1_4
    layer = Layer.build(
      number_of_neurons: 1,
      number_of_weights: 4
    )

    assert_equal(
      [[1], [1], [1], [1]],
      layer.to_matrix
    )
  end

  def test_build_layer_from_matrixx
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    layer  = Layer.from_matrix(matrix)

    assert_equal [1, 4, 7], layer.neurons[0].weights
    assert_equal [2, 5, 8], layer.neurons[1].weights
    assert_equal [3, 6, 9], layer.neurons[2].weights
  end
end
