class ForwardPropagationTest < Test::Unit::TestCase
  def test_predict
    layer_a  = Layer.build(number_of_neurons: 4, number_of_weights: 3)
    layer_b  = Layer.build(number_of_neurons: 4, number_of_weights: 4)
    layer_c  = Layer.build(number_of_neurons: 1, number_of_weights: 4)
    inputs   = [[1, 1, 0]]

    result = ForwardPropagation
      .predict([layer_a, layer_b, layer_c], inputs)
      .to_a
      .flatten
      .last
      .to_a

    assert_equal(
      [
        [
          0.9713403945491743,
          0.9713403945491743,
          0.9713403945491743,
          0.9713403945491743
        ]
      ],
      result
    )
  end
end
