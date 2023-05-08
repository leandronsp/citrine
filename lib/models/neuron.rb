# frozen_string_literal: true

module Models
  class Neuron
    attr_reader :weights

    def initialize(weights)
      @weights = weights
    end
  end
end
