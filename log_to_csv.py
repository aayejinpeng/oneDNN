import os

# 创建一个空的列表来存储每一行的数据
summary_data = []

# 获取log文件夹下的所有文件
log_folder = 'biglog'
files = os.listdir(log_folder)

# 遍历每个文件
for file_name in files:
    file_path = os.path.join(log_folder, file_name)
    
    # 检查文件的总行数是否小于10
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) < 10:
            continue  # 跳过处理该文件
    
    # 从文件名中提取信息
    parts = file_name.split('_')
    isa_name = parts[3]
    data_type = parts[2]
    thread_number = parts[-1].split('.')[0]
    network_name = parts[0]
    if(isa_name=='DEFAULT'):
        isa_name='AMX'
    
    # 从第五行开始读取文件，并将信息添加到列表中
    for line_number, line in enumerate(lines[5:], start=1):
        summary_data.append([line.strip(), data_type, thread_number,isa_name, network_name])

# 将数据写入CSV文件
with open('summary.csv', 'w') as csv_file:
    csv_file.write('Loop,Primitive,Extend info,Algorithm,Layer Type,time(ms),L1D_write,L1D_read,L1D_miss,cycles,Data Type,Thread Number,ISA,Network Name\n')  # 写入CSV文件头
    for row in summary_data:
        csv_file.write(','.join(row) + '\n')

print("Summary data saved to summary.csv")
