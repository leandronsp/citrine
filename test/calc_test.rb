class CalcTest < Test::Unit::TestCase
  def test_sigmoid
    assert_equal 0.6034832498647263, Calc.sigmoid(0.42)
  end

  def test_sigmoid_derivative
    assert_equal 0.1275, Calc.sigmoid_derivative(0.85)
  end
end
