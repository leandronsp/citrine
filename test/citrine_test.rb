require 'test/unit'
require './lib/citrine'

class CitrineTest < Test::Unit::TestCase
  def test_sigmoid
    citrine = Citrine.new

    assert_equal 0.6034832498647263, citrine.sigmoid(0.42)
  end

  def test_sigmoid_derivative
    citrine = Citrine.new

    assert_equal 0.1275, citrine.sigmoid_derivative(0.85)
  end
end
