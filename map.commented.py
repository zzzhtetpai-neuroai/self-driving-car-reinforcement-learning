# AI for Autonomous Vehicles - Build a Self-Driving Car

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Disable multitouch red dot
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Global variables
longueur = 800
largeur = 600
sand = np.zeros((longueur, largeur), dtype=np.float32)
goal_x = 20
goal_y = largeur - 20
first_update = True
last_distance = 0
last_x, last_y, n_points, length = 0, 0, 0, 0

# Deep Q-Learning setup
action2rotation = [0, 20, -20]
reward = 0

# Neural Network
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

# Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return [torch.cat(items, dim=0) for items in samples]

# Deep Q-Learning Brain
class Dqn:
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters())
        self.last_state = torch.zeros(1, input_size)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        with torch.no_grad():
            probs = F.softmax(self.model(state) * 100, dim=1)
            action = probs.multinomial(num_samples=1)
            return action.item()

    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        current_q = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        next_q = self.model(batch_next_states).detach().max(1)[0]
        target_q = batch_rewards + self.gamma * next_q
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, new_state, new_reward):
        new_state = torch.tensor(new_state, dtype=torch.float32).unsqueeze(0)
        self.memory.push((
            self.last_state,
            torch.tensor([self.last_action], dtype=torch.long),
            torch.tensor([self.last_reward], dtype=torch.float32),
            new_state
        ))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch = self.memory.sample(100)
            self.learn(*batch)
        self.last_state = new_state
        self.last_action = action
        self.last_reward = new_reward
        return action

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("no checkpoint found...")

brain = Dqn(4, 3, 0.9)

# Environment Initialization
def init():
    global sand, goal_x, goal_y, first_update
    sand = np.zeros((longueur, largeur), dtype=np.float32)
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Car and Game Classes
class Car(Widget):
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle += self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        def get_signal(sensor):
            x, y = int(sensor[0]), int(sensor[1])
            if x < 10 or x > longueur - 10 or y < 10 or y > largeur - 10:
                return 1.
            return np.sum(sand[x - 10:x + 10, y - 10:y + 10]) / 400.

        self.signal1 = get_signal(self.sensor1)
        self.signal2 = get_signal(self.sensor2)
        self.signal3 = get_signal(self.sensor3)

class Ball1(Widget): pass
class Ball2(Widget): pass
class Ball3(Widget): pass

class Game(Widget):
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):
        global brain, reward, last_distance, goal_x, goal_y, longueur, largeur, first_update

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        state = [orientation, self.car.signal1, self.car.signal2, self.car.signal3]
        action = brain.update(state, reward)
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            reward = -0.2
            if distance < last_distance:
                reward = 0.1

        for edge_check in [(self.car.x < 10, 'x', 10), (self.car.x > self.width - 10, 'x', self.width - 10),
                           (self.car.y < 10, 'y', 10), (self.car.y > self.height - 10, 'y', self.height - 10)]:
            if edge_check[0]:
                setattr(self.car, edge_check[1], edge_check[2])
                reward = -1

        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y

        last_distance = distance

class MyPaintWidget(Widget):
    def on_touch_down(self, touch):
        global last_x, last_y, n_points