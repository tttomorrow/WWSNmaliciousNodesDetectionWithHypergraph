import pandas as pd
import numpy as np
import re


def parse_log_file(log_file, time_interval):
    """
    解析日志文件，提取恶意节点、黑洞节点、丢包信息，并按时间段整理。
    返回字典，包含每个时间段内的丢包信息以及恶意和黑洞节点。
    """
    malicious_nodes = set()
    blackhole_nodes = set()
    drop_packet_info = {}

    # 解析日志文件
    with open(log_file, 'r') as file:
        for line in file:
            # 解析黑洞节点
            blackhole_match = re.match(r"Node (\d+) marked for Blackhole.", line)
            if blackhole_match:
                node_id = int(blackhole_match.group(1))
                blackhole_nodes.add(node_id)

            # 解析恶意节点（根据实际情况调整匹配规则）
            malicious_match = re.match(r"randomNumbers(\d+)", line)
            if malicious_match:
                node_id = int(malicious_match.group(1))
                malicious_nodes.add(node_id)

            # 解析丢包信息
            drop_match = re.match(r"\[node (\d+)] time:(\d+\.\d+), Drop packet (\d+)", line)
            if drop_match:
                node_id = int(drop_match.group(1))
                timestamp = float(drop_match.group(2))

                # 根据时间段分组丢包信息
                time_segment = np.floor(timestamp / time_interval) * time_interval

                if time_segment not in drop_packet_info:
                    drop_packet_info[time_segment] = {}

                if node_id not in drop_packet_info[time_segment]:
                    drop_packet_info[time_segment][node_id] = 0

                drop_packet_info[time_segment][node_id] += 1

    return {
        'malicious_nodes': malicious_nodes,
        'blackhole_nodes': blackhole_nodes,
        'drop_packet_info': drop_packet_info
    }


def process_csv(input_file, output_file, log_file, time_interval=10.0):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)
    print(df.columns)  # 查看列名

    # 解析日志文件
    log_data = parse_log_file(log_file, time_interval)

    # 确保时间列是浮动类型
    df['Time'] = df['Time'].astype(float)

    # 创建时间段列（按时间间隔分段）
    df['TimeSegment'] = np.floor(df['Time'] / time_interval) * time_interval

    # 初始化结果列表
    result = []

    # 按时间段分组，遍历每个时间段
    for time_segment, group in df.groupby('TimeSegment'):
        # 按照监听节点和被监听节点分组
        for listener, sub_group in group.groupby('listener'):
            for src_node_id, src_group in sub_group.groupby('SrcNodeId'):
                # 计算信道质量指标（例如：SNR, SignalPower, NoisePower的均值）
                avg_snr = src_group['SNR'].mean()
                avg_signal_power = src_group['SignalPower'].mean()
                avg_noise_power = src_group['NoisePower'].mean()

                # 统计每个数据包类型的发送次数
                packet_counts = src_group['PacketType'].value_counts().to_dict()

                # 获取UDP、ARP、ADOV的发送次数，其他包类型不统计
                udp_count = packet_counts.get('UDP', 0)
                arp_request_count = packet_counts.get('ARP Request', 0)
                arp_reply_count = packet_counts.get('ARP Replay', 0)
                route_request_count = packet_counts.get('Route Request', 0)
                route_reply_count = packet_counts.get('Route Replay', 0)
                route_error_count = packet_counts.get('Route Error', 0)
                route_reply_ack_count = packet_counts.get('Route Replay ACK', 0)
                ack_count = packet_counts.get('ACK', 0)


                # 获取当前时间段丢包次数
                drop_count = log_data['drop_packet_info'].get(time_segment, {}).get(src_node_id, 0)

                # 是否是恶意节点、黑洞节点
                is_malicious = 1 if src_node_id in log_data['malicious_nodes'] else 0
                is_blackhole = 1 if src_node_id in log_data['blackhole_nodes'] else 0
                is_selective_forwarding = 0  # 暂时没有选择性转发节点的具体信息

                # 创建结果记录
                result.append({
                    'ListenerNode': listener,
                    'SrcNodeId': src_node_id - 1,
                    'UDPCount': udp_count,
                    'ARPRequestCount': arp_request_count,
                    'ARPReplayCount': arp_reply_count,
                    'RouteRequestCount': route_request_count,
                    'RouteReplayCount': route_reply_count,
                    'RouteErrorCount': route_error_count,
                    'RouteReplayACKCount': route_reply_ack_count,
                    'ACKCount': ack_count,
                    'AvgSNR': avg_snr,
                    'AvgSignalPower': avg_signal_power,
                    'AvgNoisePower': avg_noise_power,
                    'PacketLossCount': drop_count,
                    'IsMaliciousNode': is_malicious,
                    'IsBlackholeNode': is_blackhole,
                    'IsSelectiveForwardingNode': is_selective_forwarding,
                    'TimeSegment': time_segment
                })

    # 将结果转换为 DataFrame
    result_df = pd.DataFrame(result)

    # 保存结果到 CSV 文件
    result_df.to_csv(output_file, index=False)