# frozen_string_literal: true

class NeuronTest < Test::Unit::TestCase
  def test_build_neuron
    neuron = Neuron.new([0.33, 0.42])

    assert_equal([0.33, 0.42], neuron.weights)
  end
end
