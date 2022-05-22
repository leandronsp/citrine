class CitrineTest < Test::Unit::TestCase
  def test_setup_layers
    citrine = Citrine.new([[1, 4], [4, 3]])

    assert_equal 2, citrine.layers.size

    assert_equal 1, citrine.layers[0].neurons.size
    assert_equal 4, citrine.layers[1].neurons.size

    assert_equal(
      [1, 1, 1, 1],
      citrine.layers[0].neurons[0].weights
    )

    assert_equal(
      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
      citrine.layers[1].neurons.map(&:weights)
    )
  end
end
