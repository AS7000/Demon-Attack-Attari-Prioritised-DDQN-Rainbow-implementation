import unittest
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.Q_Network import Q_Network  

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.input_shape = (84, 84, 4)
        self.action_size = 18
        self.q_network = Q_Network(input_shape=self.input_shape, action_size=self.action_size,length_episodes=1000,learning_rate=0.01 ,optimizer='Adam')

    def test_model_initialization(self):
        self.assertEqual(self.q_network.model.input_shape[1:], self.input_shape)
        self.assertEqual(self.q_network.model.output_shape[-1], self.action_size)

    def test_predict_output_shape(self):
        dummy_input = np.random.rand(1, *self.input_shape).astype(np.float32)
        output = self.q_network.predict(dummy_input)
        self.assertEqual(output.shape, (1, self.action_size))

    def test_model_save_and_load(self):
        dummy_input = np.random.rand(1, *self.input_shape).astype(np.float32)
        output_before = self.q_network.predict(dummy_input)

        temp_path = "unit_test.weights.h5"
        self.q_network.save(temp_path)
        self.assertTrue(os.path.exists(temp_path))

        # Create a new network and load weights
        new_q_network = Q_Network(self.input_shape, self.action_size, length_episodes=1000, learning_rate= 0.01, optimizer='Adam')
        new_q_network.load(temp_path)
        output_after = new_q_network.predict(dummy_input)

        np.testing.assert_allclose(output_before, output_after, rtol=1e-5, atol=1e-5)

        os.remove(temp_path)

    def test_model_summary(self):
        try:
            self.q_network.summary()
        except Exception as e:
            self.fail(f"Model summary failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
