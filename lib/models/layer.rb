require_relative './neuron'

module Models
  class Layer
    attr_reader :neurons

    attr_accessor :output

    def initialize(neurons)
      @neurons = neurons
    end
  end
end
