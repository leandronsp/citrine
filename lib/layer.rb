require_relative './neuron'

class Layer
  attr_reader :neurons

  def initialize(neurons)
    @neurons = neurons
  end

  def self.build(number_of_neurons:, number_of_weights:)
    number_of_neurons
      .times
      .map { [1] * number_of_weights }
      .map(&method(:build_neuron))
      .then(&method(:new))
  end

  def self.from_matrix(matrix)
    matrix
      .transpose
      .map(&method(:build_neuron))
      .then(&method(:new))
  end

  def self.build_neuron(weights) = Neuron.new(weights)

  def to_matrix = @neurons.map(&:weights).then(&:transpose)
end
