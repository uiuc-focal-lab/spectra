import os
from sender_obs import SenderMonitorInterval

MI = 75
BYTES_PER_PACKET = 1500
ALPHA = 0.025


def parse_log(file_path):
    """Parse log file and return a DataFrame."""
    data = []
    init_timestamp = None
    with open(file_path, 'r') as file:

        for line in file:
            if line.startswith('# init timestamp:'):
                init_timestamp = float(line.strip().split(': ')[1])
                init_timestamp = 0
            else:

                parts = line.split()

                timestamp = init_timestamp + float(parts[0])  # Adjust timestamp based on initial value
                if '+' in line:
                    # Ingress
                    packet_size, flow_id = int(parts[2]), int(parts[3])
                    data.append(
                        {'timestamp': timestamp, 'packet_size': packet_size, 'flow_id': flow_id, 'type': 'ingress',
                         'one_way_delay': None})

                elif '-' in line:
                    # Egress
                    packet_size, one_way_delay, flow_id = int(parts[2]), float(parts[3]), int(parts[4])
                    data.append({'timestamp': timestamp, 'packet_size': packet_size, 'one_way_delay': one_way_delay,
                                 'flow_id': flow_id, 'type': 'egress'})

    return data


def calculate_mean(data, cur_latency):
    total = cur_latency
    for item in data:
        # Format the line with the first four values from the dictionary
        total += item['cur_latency']
    ret = total / len(data)
    print(total)
    print(ret)
    return ret


def calculate_metrics(datalink_data, acklink_data, my_id):
    ret = []

    # Open the file in write mode
    # Example calculations, these need to be adjusted based on your specific definitions and data
    try:
        last_time = datalink_data[0]['timestamp']
    except:
        return None
    data_total_number = len(datalink_data)
    ack_total_number = len(acklink_data)
    cur_data_ptr = 0
    cur_ack_ptr = 0
    ret_ptr = 0
    ret.append({'latency_gradient': 0, 'latency_ratio': 0, 'sending_ratio': 0,
                'sending_rate': -1, 'cur_latency': 0, 'change_of_sending_rate': 0})
    while True:
        cur_latency = 0
        cur_sent = 0
        cur_ack = 0
        cur_sent_byte = 0
        rtt_samples = []
        if cur_data_ptr == data_total_number:
            break
        while cur_data_ptr < data_total_number and datalink_data[cur_data_ptr]['timestamp'] < last_time + MI:
            if datalink_data[cur_data_ptr]['type'] == 'ingress':
                cur_sent += 1
                cur_sent_byte += datalink_data[cur_data_ptr]['packet_size']
            if datalink_data[cur_data_ptr]['type'] == 'egress':
                cur_latency += datalink_data[cur_data_ptr]['one_way_delay']
                rtt_samples.append(datalink_data[cur_data_ptr]['one_way_delay'])
            cur_data_ptr += 1

        while cur_ack_ptr < ack_total_number and acklink_data[cur_ack_ptr]['timestamp'] < last_time + MI:
            if acklink_data[cur_ack_ptr]['type'] == 'egress':
                cur_latency += acklink_data[cur_ack_ptr]['one_way_delay']
                cur_ack += 1
                rtt_samples.append(acklink_data[cur_ack_ptr]['one_way_delay'])
            if acklink_data[cur_ack_ptr]['type'] == 'ingress':
                pass
            cur_ack_ptr += 1

        last_time = last_time + MI
        mi = SenderMonitorInterval(
            my_id,
            bytes_sent=cur_sent * BYTES_PER_PACKET,
            bytes_acked=cur_ack * BYTES_PER_PACKET,
            bytes_lost=cur_sent_byte - cur_ack * BYTES_PER_PACKET if cur_sent_byte - cur_ack * BYTES_PER_PACKET > 0 else 0,
            send_start=last_time,
            send_end=last_time + MI,
            recv_start=last_time,
            recv_end=last_time + MI,
            rtt_samples=rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

        sending_rate = cur_sent / MI

        latency_ratio = mi.get("latency ratio")
        latency_gradient = mi.get("sent latency inflation")
        sending_ratio = mi.get("send ratio")
        last_sending_rate = ret[ret_ptr - 1]['sending_rate']

        '''
                
        if last_sending_rate == 0 and sending_rate > last_sending_rate:
            change_of_sending_rate = 0
        elif sending_rate == 0 and sending_rate <= last_sending_rate:
            change_of_sending_rate = 0
        elif sending_rate > last_sending_rate:
            change_of_sending_rate = (sending_rate / last_sending_rate - 1) / ALPHA
        else:
            change_of_sending_rate = (1 - last_sending_rate / sending_rate) / ALPHA
        '''


        if sending_rate > last_sending_rate:
            change_of_sending_rate = "+"
        elif sending_rate < last_sending_rate:
            change_of_sending_rate = "="
        else:
            change_of_sending_rate = "0"

        ret.append(
            {'latency_gradient': latency_gradient, 'latency_ratio': latency_ratio, 'sending_ratio': sending_ratio,
             'sending_rate': -1, 'cur_latency': cur_latency})
        ret[ret_ptr]['sending_rate'] = sending_rate
        ret[ret_ptr]['change_of_sending_rate'] = change_of_sending_rate

        # Write the formatted line to the file

        ret_ptr += 1

    return ret


# Assuming 'ret' is your list of dictionaries with the mentioned structure

def write(data, file_path):
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the header
        file.write('latency_gradient, latency_ratio, sending_ratio, sending_rate, change_of_sending_rate\n')
        # Iterate through each dictionary in the list
        number = len(data)
        for i in range(1, number - 1):
            item = ret[i]
            # Format the line with the first four values from the dictionary
            line = f"{item['latency_gradient']}, {item['latency_ratio']}, {item['sending_ratio']}, {item['sending_rate']}, {item['change_of_sending_rate']}\n"
            # Write the formatted line to the file
            file.write(line)




algos = ['cubic', 'bbr']
gene_types = ['cloud']
gene_types = ['emulator', 'cloud']
gene_types = ['node', 'cloud']
gene_types = ['emulator']

import random
# Plz make sure that this random seed is the same as in cc environment
random.seed(0)
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
test_list = [item for item in trace_list if item not in train_list]

my_id = 0

for algo in algos:
    for gene_type in gene_types:
        k = 1
        for train_number in train_list:

            # for j in range(19):
            files = os.listdir(f'{gene_type}/{train_number}/')
            # dir_path = f'{gene_type}/2020-04-16T08-10-emu-{j}'
            for file in files:
                if k > 500:
                    break
                if '2020' not in file and '2019' not in file:
                    continue

                dir_path = f'{gene_type}/{train_number}/{file}'

                result_dir_path = f'{gene_type}_result/train'
                if not os.path.exists(result_dir_path):
                    os.makedirs(result_dir_path)

                for i in range(1, 6):

                    datalink_log_path = f'{dir_path}/{algo}_datalink_run{i}.log'
                    acklink_log_path = f'{dir_path}/{algo}_acklink_run{i}.log'
                    print(datalink_log_path)
                    # Parse logs
                    try:
                        datalink_data = parse_log(datalink_log_path)
                        acklink_data = parse_log(acklink_log_path)
                    except:
                        continue

                    # Calculate metrics
                    ret = calculate_metrics(datalink_data, acklink_data, my_id)
                    if ret is None:
                        continue
                    my_id = my_id+1
                    path = f'{result_dir_path}/{algo}_{k}.txt'
                    print(path)

                    k += 1
                    write(ret, path)
                    # print("log:", datalink_log_path)
                    # print("write:



for algo in algos:
    for gene_type in gene_types:
        k = 1
        for train_number in test_list:
            # for j in range(19):
            files = os.listdir(f'{gene_type}/{train_number}/')
            # dir_path = f'{gene_type}/2020-04-16T08-10-emu-{j}'
            for file in files:
                if k > 500:
                    break
                if '2020' not in file and '2019' not in file:
                    continue

                dir_path = f'{gene_type}/{train_number}/{file}'

                result_dir_path = f'{gene_type}_result/test'
                if not os.path.exists(result_dir_path):
                    os.makedirs(result_dir_path)

                for i in range(1, 6):

                    datalink_log_path = f'{dir_path}/{algo}_datalink_run{i}.log'
                    acklink_log_path = f'{dir_path}/{algo}_acklink_run{i}.log'
                    print(datalink_log_path)
                    # Parse logs
                    try:
                        datalink_data = parse_log(datalink_log_path)
                        acklink_data = parse_log(acklink_log_path)
                    except:
                        continue

                    # Calculate metrics
                    ret = calculate_metrics(datalink_data, acklink_data, my_id)
                    if ret is None:
                        continue
                    my_id += 1
                    path = f'{result_dir_path}/{algo}_{k}.txt'
                    print(path)

                    k += 1
                    write(ret, path)
                    # print("log:", datalink_log_path)
                    # print("write: