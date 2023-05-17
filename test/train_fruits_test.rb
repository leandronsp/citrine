# frozen_string_literal: true

class TrainFruitsTest < Test::Unit::TestCase
  def test_train_fruits_and_vegetables
    # Input Layer (4 neurons -> 3 weights)
    neuron_a = Neuron.new([-0.16595599, -0.70648822, -0.20646505])
    neuron_b = Neuron.new([0.44064899, -0.81532281, 0.07763347])
    neuron_c = Neuron.new([-0.99977125, -0.62747958, -0.16161097])
    neuron_d = Neuron.new([-0.39533485, -0.30887855, 0.370439])
    layer_a = Layer.new([neuron_a, neuron_b, neuron_c, neuron_d])

    # Hidden Layer (4 neurons -> 4 weights)
    neuron_e = Neuron.new([-0.16595599, -0.70648822, -0.20646505, -0.34093502])
    neuron_f = Neuron.new([0.44064899, -0.81532281, 0.07763347, 0.44093502])
    neuron_g = Neuron.new([-0.99977125, -0.62747958, -0.16161097, 0.14093502])
    neuron_h = Neuron.new([-0.39533485, -0.30887855, 0.370439, -0.54093502])
    layer_b = Layer.new([neuron_e, neuron_f, neuron_g, neuron_h])

    # Output Layer (1 neuron -> 4 weights)
    neuron_i = Neuron.new([-0.5910955, 0.75623487, -0.94522481, 0.64093502])
    layer_c = Layer.new([neuron_i])

    # Prepare data model
    fruits_func = -> do
      [(1..3).to_a.shuffle.take(3), 0]
    end

    vegetables_func = -> do
      [(7..9).to_a.shuffle.take(3), 1]
    end

    fruits_inputs = 50.times.map { fruits_func.call }
    vegetables_inputs = 50.times.map { vegetables_func.call }
    inputs = fruits_inputs.to_h.keys + vegetables_inputs.to_h.keys
    outputs = fruits_inputs.to_h.values + vegetables_inputs.to_h.values
    targets = Matrix[outputs].transpose.to_a

    network = NeuralNetwork.new([layer_a, layer_b, layer_c])
    network.learn(inputs, targets, 2_000)

    fruits_and_vegetables = [
      ['Apple', fruits_func.call[0]],
      ['Banana', fruits_func.call[0]],
      ['Carrot', vegetables_func.call[0]],
      ['Orange', fruits_func.call[0]],
      ['Tomato', vegetables_func.call[0]],
      ['Pineapple', fruits_func.call[0]],
      ['Potato', vegetables_func.call[0]],
      ['Cherry', fruits_func.call[0]],
      ['Garlic', vegetables_func.call[0]],
      ['Broccoli', vegetables_func.call[0]],
      ['Peach', fruits_func.call[0]],
      ['Pear', fruits_func.call[0]],
      ['Lettuce', vegetables_func.call[0]]
    ]

    fruits = %w[Apple Banana Orange Pineapple Cherry Peach Pear]
    vegetables = %w[Carrot Tomato Potato Garlic Broccoli Lettuce]

    fruits.each do |fruit|
      assert network.predict([fruits_and_vegetables.to_h[fruit]]) < 0.5
    end

    vegetables.each do |vegetable|
      assert network.predict([fruits_and_vegetables.to_h[vegetable]]) > 0.95
    end
  end
end
