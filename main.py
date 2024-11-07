import untils





input_file = 'D:/zhuomian/file/弱网中鲁棒性的恶意节点检测/exp/dataset/20241106_testTX_constantUniform_simtime-100_num_nodes-50_BHradio-0_SFradio-0_fieldLength-20_uniform-1/MonitorSnifferRx.csv'  # 输入CSV文件路径
output_file = 'D:/zhuomian/file/弱网中鲁棒性的恶意节点检测/exp/dataset/20241106_testTX_constantUniform_simtime-100_num_nodes-50_BHradio-0_SFradio-0_fieldLength-20_uniform-1/ProcessedMonitorSnifferRx.csv'  # 输出CSV文件路径
log_file = r"D:/zhuomian/file/弱网中鲁棒性的恶意节点检测/exp/dataset/20241106_testTX_constantUniform_simtime-100_num_nodes-50_BHradio-0_SFradio-0_fieldLength-20_uniform-1/log1106_uniform_20#%20_seed_12345.txt"

untils.process_csv(input_file, output_file, log_file)


