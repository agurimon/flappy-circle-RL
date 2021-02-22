from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
import time
import pickle
from datetime import date, datetime


logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 3)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    model.fit(
        X, 
        y, 
        epochs=1000,
        callbacks=[tensorboard_callback])


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=3, activation='relu')) 
    model.add(tf.keras.layers.Dense(52, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(
        loss='mse', 
        optimizer="Adam",
        metrics=['accuracy'])
    return model


class Worker():
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        self.driver = webdriver.Chrome(executable_path='./chromedriver.exe', options=self.options)
        self.driver.set_window_size(350, 820)
        self.driver.get('http://localhost:8080/medium')
        # self.driver.get('http://192.168.0.2:8080/')
        
        self.actions = ActionChains(self.driver)
        
        self.spacebar = self.actions.send_keys(Keys.SPACE)
        self.play_button = self.driver.find_element_by_css_selector('div#menu')
        self.replay_button = self.driver.find_element_by_css_selector('div#replay')
        self.sound = self.driver.find_element_by_css_selector('div#sound')
        self.sound.click()


    def get_obs(self):
        pos_now = self.driver.execute_script('return window.PosNow')
        lines = self.driver.execute_script('return window.line')
        cp = int(self.driver.execute_script('return window.CP'))

        now_line_height = -(lines[cp + 1][0] - pos_now[0]) * ((lines[cp + 1][1] - lines[cp][1]) / (lines[cp + 1][0] - lines[cp][0])) + lines[cp + 1][1]
        gap = (now_line_height - 8) - (pos_now[1] - 65)

        return np.array([pos_now[1] * 0.001, now_line_height * 0.001, gap * 0.001])

    
    def data_preparation(self, N):
        dataes = []
        for i in range(N):
            self.driver.execute_script('document.getElementById("count").innerHTML = "{}"'.format(i + 1))
            
            self.play_button.click()
            
            while True:
                jump = [1, 0]

                obs = self.get_obs()
                if obs[2] * 1000 < 20: # jump
                    jump = [0, 1]
                    self.driver.execute_script('mouseListener(new Event("none"))')
                    time.sleep(0.1)

                # if driver.execute_script('return score') > 30:
                #     time.sleep(2)

                if not self.driver.execute_script('return window.isPlaying'):
                    time.sleep(2)
                    self.replay_button.click()
                    time.sleep(1)
                    break

                dataes.append((obs, jump))
                time.sleep(0.001)

        self.driver.close()
        return dataes
    

def save(filename, dataes):
    with open(filename, 'wb') as f:
        pickle.dump(dataes, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def process(_):
    worker = Worker()
    dataes = worker.data_preparation(N)

    return dataes

N = 300
processes = 5

if __name__ == "__main__":
    filename = "30-300-1000"

    with Pool(processes=processes) as pool:
        _dataes = pool.map(process, range(processes))

        dataes = []
        for data in _dataes:
            dataes.extend(data)

        print(len(dataes))
        save("./pickle/" + filename, dataes)


    dataes = load("./pickle/" + filename)
    model = build_model()
    train_model(model, dataes)
    model.save("./model/" + filename + ".h5")