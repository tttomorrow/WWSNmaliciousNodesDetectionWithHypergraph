import csv

# 输入txt文件路径
txt_file = r'D:\file\file\AnomalyDetectionwithHypergrraph\dataset\uci\uci.txt'

# 输出csv文件路径
csv_file = r'D:\file\file\AnomalyDetectionwithHypergrraph\dataset\uci\uci.csv'

# 打开txt文件以读取数据
with open(txt_file, 'r') as txtfile:
    # 创建CSV写入器
    csv_writer = csv.writer(open(csv_file, 'w', newline=''))

    # 逐行读取txt文件，写入CSV文件
    for line in txtfile:
        # 删除末尾的换行符，并使用逗号分隔数据
        data = line.strip().split(' ')
        csv_writer.writerow(data)

print(f"{txt_file} 已成功转换为 {csv_file}")

