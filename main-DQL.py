# https://github.com/KerasKorea/KEKOxTutorial/blob/master/134_Keras%20%EC%99%80%20Gym%20%EA%B3%BC%20%ED%95%A8%EA%BB%98%ED%95%98%EB%8A%94%20Deep%20Q-Learning%20%EC%9D%84%20%ED%96%A5%ED%95%9C%20%EC%97%AC%ED%96%89.md
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

from collections      import deque
import numpy as np
import random
import os
import time


class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup      = "model/dql-medium.h5"
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95
        self.exploration_rate   = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min

        return model

    def save_model(self):
            self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class Circle():
    def __init__(self, episodes):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        self.driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=self.options)
        self.driver.set_window_size(350, 820)
        self.driver.get('http://localhost:8080/medium')
        
        self.actions = ActionChains(self.driver)
        
        self.spacebar = self.actions.send_keys(Keys.SPACE)
        self.play_button = self.driver.find_element_by_css_selector('div#menu')
        self.replay_button = self.driver.find_element_by_css_selector('div#replay')
        self.sound = self.driver.find_element_by_css_selector('div#sound')
        self.sound.click()
        
        self.before_score = 0
        
        self.sample_batch_size = 32
        self.episodes          = episodes

        self.state_size        = 8
        self.action_size       = 2
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                self.driver.execute_script('document.getElementById("count").innerHTML = "{}"'.format(index_episode + 1))
                
                score = 0
                self.before_score = 0
                self.play_button.click()
                
                state = self.get_state()
                state = np.reshape(state, [1, self.state_size])
                while True:
                    action = self.agent.act(state)
                    
                    next_state, reward, done = self.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    
                    if done:
                        time.sleep(1)
                        try:
                            self.replay_button.click()
                        except:
                            time.sleep(1)
                            self.replay_button.click()
                        break
                    
                print("Episode {}# Score: {}".format(index_episode, score))
                self.agent.replay(self.sample_batch_size)
        
        finally:
            self.agent.save_model()            
            
    def step(self, action): 
        if action:
            self.play_button.click()
            time.sleep(0.4)

        now_score = self.driver.execute_script('return score')
        score = 1.0 if self.before_score < now_score else 0
        self.before_score = now_score
        
        is_playing = self.driver.execute_script('return window.isPlaying')
        done = False if is_playing else True
        state = self.get_state()
        
        return state, score, done
    
    def get_state(self):
        pos_now = self.driver.execute_script('return window.PosNow')
        lines = self.driver.execute_script('return window.line')
        CP = int(self.driver.execute_script('return window.CP'))
        now_line_height = -(lines[CP + 1][0] - pos_now[0]) * ((lines[CP + 1][1] - lines[CP][1]) / (lines[CP + 1][0] - lines[CP][0])) + lines[CP + 1][1]
        gap = (now_line_height - 8) - (pos_now[1] - 65)
        vx = self.driver.execute_script('return window.Vx')
        
        return np.array([
            pos_now[1] * 0.001, 
            lines[CP + 0][1] * 0.001, 
            lines[CP + 1][1] * 0.001, 
            lines[CP + 2][1] * 0.001,
            lines[CP + 2][1] * 0.001,
            now_line_height * 0.001, 
            gap * 0.1,
            vx * 0.001])


if __name__ == "__main__":
    circle = Circle(10000)
    circle.run()