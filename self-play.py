from numpy.core.numerictypes import _construct_lookups
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool

import tensorflow as tf
import numpy as np

import random
import time


class Worker():
    def __init__(self):
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


    def data_preparation(self, N, f, a=False):
        game_data = []

        for count in range(N):
            # 몇번째 돌아가는지 체크
            self.driver.execute_script('document.getElementById("count").innerHTML = "{}"'.format(count + 1))
            game_steps = []
            score = 0
            
            self.play_button.click()
            
            self.before_score = 0

            while True:
                obs = self.get_obs()
                action = f(obs)

                reward, done = self.step(action)
                game_steps.append((obs, action))

                score += reward
                
                if done:
                    time.sleep(2)
                    self.replay_button.click()
                    time.sleep(1)
                    break
            
                time.sleep(0.001)

            game_data.append((score, game_steps))

        return game_data

    
    def step(self, action): 
        if action:
            self.play_button.click()
            time.sleep(0.1)

        now_score = self.driver.execute_script('return score')
        score = 1 if self.before_score < now_score else 0
        self.before_score = now_score
        
        is_playing = self.driver.execute_script('return window.isPlaying')
        done = False if is_playing else True
        
        return score, done

    
    def get_obs(self):
        pos_now = self.driver.execute_script('return window.PosNow')
        lines = self.driver.execute_script('return window.line')
        CP = int(self.driver.execute_script('return window.CP'))
        now_line_height = -(lines[CP + 1][0] - pos_now[0]) * ((lines[CP + 1][1] - lines[CP][1]) / (lines[CP + 1][0] - lines[CP][0])) + lines[CP + 1][1]
        gap = (now_line_height - 8) - (pos_now[1] - 65)

        return np.array([pos_now[1] * 0.001, now_line_height * 0.001, gap * 0.001])


def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 3)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    model.fit(X, y, epochs=1000)


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=3, activation='relu')) 
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(loss='mse', optimizer="Adam")
    return model


def mfilter(game_dataes, K):
    game_data = []
    for data in game_dataes:
        game_data.extend(data)

    game_data.sort(key=lambda s:-s[0])

    training_set = []
    for i in range(K):
        print(game_data[i][0], end=", ")
        for stepp in game_data[i][1]:

            if stepp[1] == 0: # no jump
                training_set.append((stepp[0], [1, 0]))
            else:
                training_set.append((stepp[0], [0, 1]))
    
    return training_set


def process(_):
    # model = tf.keras.models.load_model('sibal.h5')
    worker = Worker()
    training_data = worker.data_preparation(N, lambda s: random.randrange(0, 2))

    worker.driver.close()

    return training_data


def proro(_):
    def predictor(s):
        pre = model.predict(s.reshape(-1, 3))[0]
        a = np.random.choice([0, 1], p=pre)
        return a
    
    worker = Worker()
    training_data = worker.data_preparation(N, predictor)
    worker.driver.close()

    return training_data


model = build_model()
processes = 6
N = 1000
K = 300
self_playing = 7

if __name__ == '__main__':
    with Pool(processes=processes) as pool:
        game_dataes = pool.map(process, range(processes))
    
    training_set = mfilter(game_dataes, K)
    train_model(model, training_set)
    model.save("210218-3.h5")
    
    for i in range(self_playing):
        with Pool(processes=processes) as pool:
            game_dataes = pool.map(proro, range(processes))

        training_set = mfilter(game_dataes, K)
        train_model(model, training_set)

    model.save("210218-4.h5")


# if __name__ == '__main__':
#     model = tf.keras.models.load_model('model/self-playing-medium.h5')
#     def predictor(s):
#         s[2] = s[2] - 0.015
#         pre = model.predict(s.reshape(-1, 3))[0]
#         a = np.random.choice([0, 1], p=pre)
#         print(s)
#         print(pre)
#         print(a)
#         return a
    
#     worker = Worker()
#     worker.data_preparation(100, predictor)
#     worker.driver.close()