# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import heapq
import time
import random
import json
import os
import sys
import inspect

random.seed(0)
np.random.seed(0)


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common import sender_obs, config
from common.simple_arg_parse import arg_or_default

K = 10

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

MAX_STEPS = 400

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

USE_CWND = False

BANDWIDTH_Mbps = 0.57 * 1000
ONE_WAY_DELAY_ms = 14 / 1000
PACKET_SIZE_BYTES = 1500  # Standard Ethernet packet size
LOSS_RATE = 0.000477  # Packet loss rate
QUEUE_SIZE = 14  # Number of packets

TRACE_LENGTH = 121


num_to_trace_map = {
    "1": "0.57mbps-poisson.trace",
    "2": "2.64mbps-poisson.trace",
    "3": "3.04mbps-poisson.trace",
    "4": "100.42mbps.trace",
    "5": "77.72mbps.trace",
    "6": "114.68mbps.trace",
    "7": "12mbps.trace",
    "8": "60mbps.trace",
    "9": "108mbps.trace",
    "10": "12mbps.trace",
    "11": "60mbps.trace",
    "12": "108mbps.trace",
    "13": "0.12mbps.trace",
    "14": "10-every-200.trace",
    "15": "12mbps.trace",
    "16": "12mbps.trace",
    "17": "12mbps.trace",
    "18": "12mbps.trace"
}




trace_list = list(num_to_trace_map.keys())
# Calculate 75% of the list size for the training set
train_size = int(0.75 * len(trace_list))

# Sample 75% of the elements for the training set
train_list = random.sample(trace_list, train_size)

# Get the test set by taking elements not in the training set
test_list = [item for item in trace_list if item not in train_list]

num_to_delay = {
    "1": 28,
    "2": 88,
    "3": 130,
    "4": 27,
    "5": 51,
    "6": 45,
    "7": 10,
    "8": 10,
    "9": 10,
    "10": 50,
    "11": 50,
    "12": 50,
    "13": 10,
    "14": 10,
    "15": 30,
    "16": 30,
    "17": 30,
    "18": 30
}

num_to_loss = {
    "1": 0.0477,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    "10": 0,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
}

num_to_queue_size = {
    "1": 14,
    "2": 130,
    "3": 426,
    "4": 173,
    "5": 94,
    "6": 450,
    "7": 1,
    "8": 1,
    "9": 1,
    "10": 1,
    "11": 1,
    "12": 1,
    "13": 10000,
    "14": 10000,
    "15": 6,
    "16": 20,
    "17": 30,
    "18": 60
}


class Link():
    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = bandwidth  # Bandwidth in Mbps
        print(self.bw)

        self.dl = delay  # One-way delay in ms
        self.lr = loss_rate  # Loss rate as a decimal
        self.queue = []  # Queue to hold packets


        self.max_queue_delay = (queue_size * PACKET_SIZE_BYTES * 8) / (bandwidth * 1e6)  # Maximum delay due to queue

    def packet_enters_link(self, event_time):
        if random.random() < self.lr:
            return False
        if len(self.queue) >= QUEUE_SIZE:
            return False  # Drop the packet if the queue is full
        self.queue.append(event_time)  # Simulate packet entering the queue
        return True

    def update_queue(self, current_time):
        while self.queue and current_time - self.queue[0] > self.max_queue_delay:
            self.queue.pop(0)  # Remove packets that have 'left' the queue

    def get_cur_latency(self, current_time):
        self.update_queue(current_time)
        return self.dl + (len(self.queue) * PACKET_SIZE_BYTES * 8) / (self.bw * 1e6)

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0


'''
class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw

    def get_cur_queue_delay(self, event_time):
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        #print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            #print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        #print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

'''


class Network():

    def __init__(self, senders, links):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False))

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            # print("Got event %s, to link %d, latency %f at time %f" % (event_type, next_hop, cur_latency, event_time))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                        # print("Packet lost at time %f" % self.cur_time)
                    else:
                        sender.on_packet_acked(cur_latency)
                        # print("Packet acked at time %f" % self.cur_time)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    # print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q,
                                   (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)

            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")

        reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)

        return reward * REWARD_SCALE


class Sender():

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

    _next_id = 1

    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        # print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()

        # obs_dur = obs_end_time - self.obs_start_time
        # print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        # print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        # print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        # print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)


class SimulatedNetworkEnv(gym.Env):

    def __init__(self,
                 train=True,
                 history_len=arg_or_default("--history-len", default=K),
                 features=arg_or_default("--input-features",
                                         default="sent latency inflation,"
                                                 + "latency ratio,"
                                                 + "send ratio")):
        self.viewer = None
        self.rand = None

        self.train = True

        self.min_bw, self.max_bw = (100, 500)
        self.min_lat, self.max_lat = (0.05, 0.5)
        self.min_queue, self.max_queue = (0, 8)
        self.min_loss, self.max_loss = (0.0, 0.05)
        self.history_len = history_len

        self.features = features.split(",")


        self.trace_data = self.load_trace_data()
        self.current_trace_index = 0

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = MAX_STEPS
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        # action space
        if USE_CWND:
            self.action_space = spaces.Box(np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.rewards = []

        # observation space
        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)
        print('features: ', self.features)
        print('observation space min:', single_obs_min_vec)
        print('observation space max:', single_obs_max_vec)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events": []}
        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def load_trace_data(self):
        trace_data = []
        if self.train:
            use_list = train_list
        else:
            use_list = test_list
        for train_file in use_list:
            file_name = f"pantheon-traces/bandwidth_{num_to_trace_map[train_file]}"

            with open(file_name, 'r') as file:
                lines = file.readlines()

            num_lines = len(lines)
            full_repeats = TRACE_LENGTH // num_lines
            partial_repeat_count = TRACE_LENGTH % num_lines

            for _ in range(full_repeats):
                for line in lines:
                    parts = line.strip().split()
                    bw = float(parts[0])
                    if bw==0:
                        continue
                    trace_data.append((bw,
                                       num_to_delay[train_file],
                                       num_to_queue_size[train_file],
                                       num_to_loss[train_file]))

                # Add the partial repeats (if any)
            for line in lines[:partial_repeat_count]:
                parts = line.strip().split()
                bw = float(parts[0])
                if bw == 0:
                    continue
                trace_data.append((bw,
                                   num_to_delay[train_file],
                                   num_to_queue_size[train_file],
                                   num_to_loss[train_file]))

        return trace_data

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1, )
        # print(sender_obs)
        return sender_obs

    def step(self, actions):
        # print("Actions: %s" % str(actions))
        # print(actions)
        # print('pantheon env')
        for i in range(0, 1):  # len(actions)):
            # print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if USE_CWND:
                self.senders[i].apply_cwnd_delta(action[1])
        # print("Running for %fs" % self.run_dur)
        reward = self.net.run_for_dur(self.run_dur)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        sender_mi = self.senders[0].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        # event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        # event["Cwnd"] = sender_mi.cwnd
        # event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        if event["Latency"] > 0.0:
            self.run_dur = 0.5 * sender_mi.get("avg latency")
        # print("Sender obs: %s" % sender_obs)

        should_stop = False

        self.reward_sum += reward
        return sender_obs, reward, (self.steps_taken >= self.max_steps or should_stop), {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw, delay, queue, loss = self.trace_data[self.current_trace_index]
        delay = delay/1000
        bw = bw*1000




        self.current_trace_index = self.current_trace_index + 1 if self.current_trace_index < len(
            self.trace_data) - 1 else 0

        link = Link(bw, delay, queue, loss)
        self.links = [link]  # Simplified to one link for demonstration
        self.senders = [Sender(bw * 0.5, [link], 0, self.features, history_len=self.history_len)]

        self.run_dur = 3* delay

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1
        #if self.episodes_run > 0 and self.episodes_run % 100 == 0:
        #    self.dump_events_to_file("pcc_env_log_run_%d.json" % self.episodes_run)
        self.event_record = {"Events": []}
        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        self.rewards.append(self.reward_ewma)
        if (self.episodes_run + 1) % 100 == 0:
            np.save("records_k%d.npy" % K, np.array(self.rewards))
        # print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.event_record, f, indent=4)


register(id='PccNs-v0-pantheon', entry_point='network_sim:SimulatedNetworkEnv')
# env = SimulatedNetworkEnv()
# env.step([1.
