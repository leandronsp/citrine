# frozen_string_literal: true

require 'byebug'
require 'test/unit'

Dir['./lib/**/*.rb'].sort.each { |file| require file }
#Dir['./test/**/*.rb'].sort.each { |file| require file }
