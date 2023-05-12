require_relative './neuron'

module Models
  class Layer
    attr_reader :neurons

    def initialize(neurons)
      @neurons = neurons
    end

    def to_matrix
      neurons.map(&:weights).then(&:transpose)
    end
  end
end
