# frozen_string_literal: true

require 'matrix'

class MatrixText < Test::Unit::TestCase
  def test_standard_matrix_multiply
    assert_equal(
      [[2, 2, 2], [4, 4, 4], [6, 6, 6]],
      (Matrix[[1], [2], [3]] * Matrix[[2, 2, 2]]).to_a
    )
  end

  def test_standard_matrix_add
    assert_equal(
      [[2, 3, 4]],
      (Matrix[[1, 2, 3]] + Matrix[[1, 1, 1]]).to_a
    )
  end

  def test_standard_matrix_subtract
    assert_equal(
      [[1, 2, 3]],
      (Matrix[[2, 3, 4]] - Matrix[[1, 1, 1]]).to_a
    )
  end

  def test_standard_matrix_transpose
    assert_equal(
      [[0], [1], [0], [1]],
      (Matrix[[0, 1, 0, 1]].transpose).to_a
    )
  end
end
