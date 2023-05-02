# frozen_string_literal: true

class BackPropagationTest < Test::Unit::TestCase
  def test_delta
    delta = BackPropagation.delta([[1, 0, 1]], [[0.42, 0.31, 0.12]])

    assert_equal [
      [0.14128800000000002, -0.06630899999999999, 0.092928]
    ], delta
  end

  def test_delta_middle_layer
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    output_layer  = Layer.from_matrix(matrix)

    delta = BackPropagation.delta_middle_layer(
      [[9, 9, 9]],
      output_layer,
      [[4, 3, 8]]
    )

    assert_equal [
      [-2448, -5688, -8928]
    ], delta
  end
end
