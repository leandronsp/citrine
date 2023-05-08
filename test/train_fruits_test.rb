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
      [(1..20).to_a.shuffle.take(3), 0]
    end

    vegetables_func = -> do
      [(21..40).to_a.shuffle.take(3), 1]
    end

    fruits_inputs = 50.times.map { fruits_func.call }
    vegetables_inputs = 50.times.map { vegetables_func.call }

    inputs = fruits_inputs.to_h.keys + vegetables_inputs.to_h.keys
    outputs = fruits_inputs.to_h.values + vegetables_inputs.to_h.values
    targets = Matrix[outputs].transpose.to_a

    network = NeuralNetwork.new([layer_a, layer_b, layer_c])
    network.learn(inputs, targets, 2_000)

    is_fruit = -> { fruits_func.call[0] }
    is_vegetable = -> { vegetables_func.call[0] }

    fruits_and_vegetables = [
      ['Apple', is_fruit.call],
      ['Banana', is_fruit.call],
      ['Carrot', is_vegetable.call],
      ['Orange', is_fruit.call],
      ['Tomato', is_vegetable.call],
      ['Pineapple', is_fruit.call],
      ['Potato', is_vegetable.call],
      ['Cherry', is_fruit.call],
      ['Garlic', is_vegetable.call],
      ['Broccoli', is_vegetable.call],
      ['Peach', is_fruit.call],
      ['Pear', is_fruit.call],
      ['Lettuce', is_vegetable.call]
    ]

    fruits = %w[Apple Banana Orange Pineapple Cherry Peach Pear]
    vegetables = %w[Carrot Tomato Potato Garlic Broccoli Lettuce]

    fruits.each do |fruit|
      assert network.predict!([fruits_and_vegetables.to_h[fruit]]) < 0.5
    end

    vegetables.each do |vegetable|
      assert network.predict!([fruits_and_vegetables.to_h[vegetable]]) > 0.95
    end
  end
end
