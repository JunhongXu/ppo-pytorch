from carla_env.carla.client import CarlaClient
import numpy as np
import re
import subprocess
from gym.spaces import Box, Tuple, Discrete
import cv2
from collections import namedtuple
# TODO: Add observations of pedestrians, traffic lights, and vehicle in the field of view.
Action = namedtuple('Action', 'throttle brake steer')


class CarlaEnv(object):
    def __init__(self, config_file, host='localhost', port=2000, throttle=True,
                 brake=True, reverse=False, town_id='01', resX=600, resY=400):
        # open up carla using subprocess
        # cmd = '~/CARLA/CarlaUE4.sh /Game/Maps/Town{} -benchmark -fps=15 -windowed -ResX={} -ResY={}'\
        #     .format(town_id, resX, resY)
        # subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # wait 4 seconds till CARLA starts
        # time.sleep(4)
        self.carla = CarlaClient(host, port, timeout=10000)
        self.carla.connect()

        self.max_speed = 40 # this can be set
        self.terminal_judge_start = 400
        self.time_step = 0
        self.current_speed = 0
        self.current_throttle = 0
        self.current_brake = 0

        self.config_file = config_file
        with open(self.config_file) as f:
            self.config_file = f.read()
        self.throttle = throttle
        self.brake = brake
        self.reverse = reverse

        self.action_space = self.__create_action_space()
        self.obs_space = self.__create_obs_space()

    def __create_obs_space(self):
        x, y = 224, 224
        rgb = 'RGB'
        labels = 'Labels'
        depth = 'DepthMap'
        obs_space = None
        if rgb and labels and depth:
            # rgb, depth, and semantic
            obs_space = Tuple((Box(0, 255, shape=(x, y, 3)), Box(0, 1000, shape=(x, y, 1)), Box(0, 12, shape=(x, y, 1))))
        elif rgb and labels and not depth:
            # rgb and semantic
            obs_space = Tuple((Box(0, 255, shape=(x, y, 3)), Box(0, 12, shape=(x, y, 1))))
        elif rgb and not labels and not depth:
            # rgb only
            obs_space = Box(0, 255, shape=(x, y, 3))
        elif rgb and not labels and depth:
            # rgb, depth
            obs_space = Tuple((Box(0, 255, shape=(x, y, 3)), Box(0, 12, shape=(x, y, 1))))
        elif depth and not rgb and not labels:
            # depth
            obs_space = Box(0, 1, shape=(x, y, 1))
        elif depth and not rgb and labels:
            # depth and semantic
            obs_space = Tuple((Box(0, 1000, shape=(x, y, 1)), Box(0, 12, shape=(x, y, 1))))
        elif not depth and not rgb and labels:
            # semantic only
            obs_space = Box(0, 12, shape=(x, y, 1))

        return obs_space

    def __create_action_space(self):
        action_space = None
        if self.throttle and self.brake and self.reverse:
            # steer throttle reverse and brake
            action_space = Tuple((Box(-1, 1, shape=(3, )), Discrete(1)))
        elif self.throttle and self.brake and not self.reverse:
            # steer, throttle, and brake
            action_space = Box(-1, 1, shape=(3,))
        elif self.throttle and not self.brake and not self.reverse:
            # steer, throttle
            action_space = Box(-1, 1, shape=(2, ))
        elif self.throttle and not self.brake and self.reverse:
            # steer, throttle, reverse
            action_space = Tuple((Box(-1, 1, shape=(2, )), Discrete(1)))
        elif not self.throttle and not self.brake and not self.reverse:
            # steer
            action_space = Box(-1, 1, shape=(1, ))
        elif self.brake and not self.throttle and not self.reverse:
            # steer, brake
            action_space = Box(-1, 1, shape=(2, ))
        elif self.brake and not self.throttle and self.reverse:
            # steer, brake, reverse
            action_space = Tuple((Box(-1, 1, shape=(2, )), Discrete(1)))
        elif not self.brake and not self.throttle and self.reverse:
            # steer, reverse
            action_space = Tuple((Box(-1, 1, shape=(1, )), Discrete(1)))
        return action_space

    def __make_observations(self, measurements, sensor_data):
        """
        returns a dictionary of :
            1. image observation (including rgb, semantic segmentation, and depth if enabled)
            2. agent orientation (x, y, z)
            3. agent current speed (x)
            4. agent current acceleration (x, y, z)
            5. agent collision with pedestrians, vehicles, and other objects
        """
        rgb = sensor_data['RGB'].data
        depth = sensor_data['Depth'].data
        labels = sensor_data['Semantic'].data

        img_dict = {}
        if len(rgb) > 0:
            img_dict['RGB'] = rgb
        if len(depth):
            img_dict['Depth'] = depth
        if len(labels) > 0:
            img_dict['Labels'] = labels

        player_measurements = measurements.player_measurements
        orientation = np.array([
            player_measurements.transform.orientation.x,
            player_measurements.transform.orientation.y,
            player_measurements.transform.orientation.z
        ])

        location = np.array([
            player_measurements.transform.location.x,
            player_measurements.transform.location.y,
            player_measurements.transform.location.z
        ])

        acceleration = np.array([
            player_measurements.acceleration.x,
            player_measurements.acceleration.y,
            player_measurements.acceleration.z
        ])

        speed = float(player_measurements.forward_speed)   # [Max ~90?] forward speed?
        collision_vehicle = float(player_measurements.collision_vehicles)  # [??]
        collision_pedestrians = float(player_measurements.collision_pedestrians)   # [??]
        collision_other = float(player_measurements.collision_other)   # [??]
        intersection_otherlane = float(player_measurements.intersection_otherlane) # [0-1]
        intersection_offroad = float(player_measurements.intersection_offroad) # [0-1]
        self.current_speed = speed
        return {
            'image': img_dict,
            'orientation': orientation,
            'location': location,
            'acceleration': acceleration,
            'speed': speed,
            'collision_vehicle': collision_vehicle,
            'collision_ped': collision_pedestrians,
            'collision_other': collision_other,
            'intersection_otherlane': intersection_otherlane,
            'intersection_offroad': intersection_offroad
        }

    def __update_action(self, action):
        steer = action[0]

        if self.throttle:
            throttle = action[1]
        else:
            # simple throttle control if throttle is disabled
            if self.current_speed < self.max_speed:
                # accelerate
                self.current_throttle += 0.01
            throttle = self.current_throttle
            action = np.insert(action, 1, self.current_throttle)
#            action.insert(1, self.current_throttle)

        if self.brake:
            brake = action[2]
        else:
            if self.current_speed > self.max_speed:
                self.current_brake += 0.01
            brake = self.current_brake
            action = np.insert(action, 2, self.current_brake)
        if self.reverse:
            reverse = action[3]

        agent_action = Action(throttle=throttle, steer=steer, brake=brake)
        return agent_action

    def reset(self):
        self.scene = self.carla.load_settings(self.config_file)
        self.carla.start_episode(np.random.randint(0, len(self.scene.player_start_spots) - 1))
        measurements, sensor_data = self.carla.read_data()
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.time_step = 0
        obs_dict = self.__make_observations(measurements, sensor_data)
        return obs_dict

    def step(self, action):
        """
        action should be in the format of
        (steer, acceleration(opt), brake(opt), reverse(opt))
        """
        # TODO: Add more complex reward function and terminal criterion
        agent_action = self.__update_action(action)
        self.carla.send_control(
            steer=agent_action.steer,
            throttle=agent_action.throttle,
            brake=agent_action.brake,
            hand_brake=False,
            reverse=False
        )
        measurements, sensor_data = self.carla.read_data()
        obs_dict = self.__make_observations(measurements, sensor_data)

        # reward is the current speed
        reward = max(0, self.current_speed)
        done = False

        # episode terminates if the car collides into objectrs
        collision = obs_dict['collision_vehicle'] \
                    + obs_dict['collision_ped'] + \
                    obs_dict['collision_other']
        if collision > 0:
            done = True
            reward = -1

        # episode terminates if the car violates traffic rule
        intersection = obs_dict['intersection_otherlane'] + obs_dict['intersection_offroad']
        if intersection > 0:
            done = True
            reward = -1

        # episode terminates if the car barely moves
        progress = self.current_speed
        if self.terminal_judge_start < self.time_step:
            if progress < 5:    # 5kmh
                done = True
                reward = -1
        self.time_step += 1
        return obs_dict, reward, done, {}


if __name__ == '__main__':
    carla = CarlaEnv('CarlaSettings.ini', throttle=False, brake=False)
    obs_dict = carla.reset()
    for i in range(0, 100):
        action = carla.action_space.sample()
        obs, reward, done, _ = carla.step(action)
        cv2.imshow('depth', obs['image']['Depth'])
        cv2.waitKey()

    carla.carla.disconnect()