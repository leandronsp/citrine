# frozen_string_literal: true

module ForwardPropagation
  class Forward
    def self.call(*args) = new(*args).call

    def initialize(layers, inputs)
      @layers = layers
      @inputs = inputs
    end

    def call
      @layers.map do |layer|
        layer.tap { layer.output = predict(layer) }
      end
    end

    private

    def predict(layer)
      (Matrix[*@inputs] * Matrix[*layer.to_matrix])
        .map(&Utils::ActivationFunction.method(:sigmoid))
    end
  end
end
