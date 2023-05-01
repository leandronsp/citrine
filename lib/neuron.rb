# frozen_string_literal: true

class Neuron
  attr_reader :weights

  def initialize(weights)
    @weights = weights
  end
end
