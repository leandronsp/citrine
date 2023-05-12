# frozen_string_literal: true

require_relative './forward'

module ForwardPropagation
  class Predict
    def self.call(*args) = new(*args).call

    def initialize(layers, inputs)
      @layers = layers
      @inputs = inputs
    end

    def call
      Forward.call(@layers, @inputs).last.output
    end
  end
end
