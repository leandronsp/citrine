# frozen_string_literal: true

class NaiveMatrixTest < Test::Unit::TestCase
  def test_naive_matrix_multiply
    assert_equal(
      [[2, 4, 6]],
      NaiveMatrix.new([[1, 2, 3]], [[2, 2, 2]]).multiply
    )
  end
end
