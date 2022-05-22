require 'test/unit'
require './lib/naive_matrix'
require 'matrix'

class NaiveMatrixTest < Test::Unit::TestCase
  def test_naive_matrix_multiply
    assert_equal(
      [[2, 4, 6]],
      NaiveMatrix.new.multiply([[1, 2, 3]], [[2, 2, 2]])
    )
  end

  def test_standard_matrix_multiply
    assert_equal(
      [[2, 2, 2], [4, 4, 4], [6, 6, 6]],
      (Matrix[[1], [2], [3]] * Matrix[[2, 2, 2]]).to_a
    )
  end
end
