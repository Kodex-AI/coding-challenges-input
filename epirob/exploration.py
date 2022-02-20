# author: Claus Lang
# email: claus.lang@bccn-berlin.de
import naoqi
import time
import util
import traceback
import datetime
import cPickle
import gzip
import numpy as np
from images import ImageRetrieval
from util import load_state_memory


class Exploration:
    
    def __init__(self, ip=None, port=None, fps=30):
        self.ip = ip if ip is not None else util.IP
        self.port = port if port is not None else util.PORT
        self.fps = fps
        self.joints = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
        self.image_retrieval = ImageRetrieval(fps=self.fps)
        self.state_memory = []
        self.speech_proxy = naoqi.ALProxy('ALTextToSpeech', self.ip, self.port)
        self.motion_proxy = naoqi.ALProxy('ALMotion', self.ip, self.port)
        self.motion_proxy.rest()
        self.wait_until_done(lambda *args: None)
        self.motion_proxy.setStiffnesses('Body', 1.0)
        self.motion_proxy.setAngles('HeadYaw', 0.3, 0.2)
        self.wait_until_done(lambda *args: None)

    def relax(self):
        self.motion_proxy.setStiffnesses('Body', 0.0)

    def wait_until_done(self, action, interval_time=0.1, **kwargs):
        new_angles = self.motion_proxy.getAngles(self.joints, False)
        old_angles = None
        while not np.array_equal(old_angles, new_angles):
            action(**kwargs)
            time.sleep(interval_time)
            old_angles = new_angles
            new_angles = self.motion_proxy.getAngles(self.joints, False)

    def explore_random(self, num_positions, speed=0.3, pause_length=1):
        # noinspection PyBroadException
        try:
            for i in range(num_positions):
                print('trial ' + str(i))
                time.sleep(pause_length)
                intended_angles = [np.random.uniform(util.RANGES[joint]['min'], util.RANGES[joint]['max'])
                                   for joint in self.joints]
                self.motion_proxy.setAngles(self.joints, intended_angles, speed)
                self.wait_until_done(self.capture_state, interval_time=1. / self.fps, counter=i)
        except Exception:
            print traceback.format_exc()

        time.sleep(pause_length)
        self.motion_proxy.rest()
        self.wait_until_done(lambda *args: None)
        self.motion_proxy.setStiffnesses('Body', 0.0)

    def explore_from_to(self, start_angles, end_angles, speed=0.3):
        self.motion_proxy.setAngles(self.joints, start_angles, speed)
        self.wait_until_done(lambda *args: None, interval_time=1. / self.fps)
        self.speech_proxy.say('Start exploring')
        self.motion_proxy.setAngles(self.joints, end_angles, speed)
        self.wait_until_done(self.capture_state, interval_time=1. / self.fps, counter=0)

    def capture_state(self, counter):
        command_angles = self.motion_proxy.getAngles(self.joints, False)
        sensor_angles = self.motion_proxy.getAngles(self.joints, True)
        image = self.image_retrieval.capture()
        if image is None:
            print('WARNING! Image is None!')
            self.speech_proxy.say('WARNING! Image is None!')
        else:
            self.state_memory.append({'command_angles': command_angles, 'sensor_angles': sensor_angles,
                                      'image': image, 'counter': counter})

    def save_state_memory(self, name, timestamp=True):
        if timestamp:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%X')
            file_name = 'data/images/' + name + '_' + timestamp + '.pkl'
        else:
            file_name = 'data/images/' + name + '_original.pkl'
        with gzip.open(file_name, 'wb') as memory_file:
            cPickle.dump(self.state_memory, memory_file, protocol=cPickle.HIGHEST_PROTOCOL)
        return file_name


def generate_agency_trials(index):
    joints = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
    start_angles = [np.random.uniform(util.RANGES[joint]['min'], util.RANGES[joint]['max'])
                    for joint in joints]
    end_angles = [np.random.uniform(util.RANGES[joint]['min'], util.RANGES[joint]['max'])
                  for joint in joints]
    file_names = []

    for trial in ('a', 'b'):
        exploration = Exploration(fps=30)

        start_time = datetime.datetime.now()
        print 'started at', start_time
        exploration.explore_from_to(start_angles, end_angles, speed=0.1)
        end_time = datetime.datetime.now()
        print 'exploration took', end_time - start_time

        start_time = datetime.datetime.now()
        print 'saving started at', start_time
        file_name = exploration.save_state_memory('agency_new/agency_{}{}'.format(index, trial), timestamp=False)
        file_names.append(file_name)
        end_time = datetime.datetime.now()
        print 'saving took', end_time - start_time

    exploration.motion_proxy.rest()
    exploration.wait_until_done(lambda *args: None)
    exploration.motion_proxy.setStiffnesses('Body', 0.0)
    return file_names


def main():

    exploration = Exploration(fps=30)
    exploration.explore_random(1, speed=0.1)
    return

    start_time = datetime.datetime.now()
    print 'started at', start_time
    exploration.explore_random(5, speed=0.1)
    end_time = datetime.datetime.now()
    print 'exploration took', end_time - start_time

    start_time = datetime.datetime.now()
    print 'saving started at', start_time
    file_name = exploration.save_state_memory('eval/test')
    end_time = datetime.datetime.now()
    print 'saving took', end_time - start_time

    for memory in load_state_memory(file_name):
        ImageRetrieval.show(memory['image'])


if __name__ == '__main__':
    main()
