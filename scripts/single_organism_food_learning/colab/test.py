import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gymnasium as gym
from env import CustomEnv

class DQNTester:
    def __init__(self, model_path, env):
        self.model = load_model(model_path)
        self.env = env

    def test_agent(self, episodes=5):
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, 240, 320, 3])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, 240, 320, 3])
                state = next_state
                score += reward

            print(f"Episode: {episode+1}, Score: {score}")

        self.env.close()

if __name__ == "__main__":
    env = CustomEnv()  # CustomEnv sınıfınızı burada başlatın
    tester = DQNTester('dqn_model_final.h5', env)
    tester.test_agent()
