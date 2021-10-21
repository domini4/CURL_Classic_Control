# -*- coding: utf-8 -*-
# MIT License
#
# Copyright (c) 2017 Kai Arulkumaran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# ==============================================================================
from collections import deque
import random
import atari_py
import cv2
import torch
import gym
from skimage import color
from skimage import io


class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    print('atari actions', actions)
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    print('atari self.actions', self.actions)
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode
    # gym modifications
    self.env_name = "CartPole-v1"
    self.env1 = gym.make(self.env_name)
    print('observation space:', self.env1.observation_space.shape)
    print('action space:', self.env1.action_space)

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def rendor_env(self):
    for episode in range(10):
        self.env1.reset()
        for t in range(500):
            img = self.env1.render(mode='rgb_array')
            #print(color.rgb2gray(img).shape)
            img_new = cv2.resize(color.rgb2gray(img), (126, 84), interpolation = cv2.INTER_CUBIC)
            #print(img_new.shape)
            action = self.env1.action_space.sample()
            next_state, reward, done, info = self.env1.step(action)
            if done:
                break
            cv2.imshow("Resized image", img_new)

  def reset(self):
    #print('in reset')
    if self.life_termination:
      print(' in if')
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      print('in else')
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    print('outside')
    observation = self._get_state()
    #print('james state')
    #torch.set_printoptions(threshold=10_000)
    #print(observation)
    self.state_buffer.append(observation)
    self.lives = self.ale.lives() # do not need for gym
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action)) # pass in the action to get reward
      #print('reward:', self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over() # done or not
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    # no need to set lives in gym
    if self.training:
      lives = self.ale.lives() # no need in gym
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert # only for DMControl
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()
