def get_last_two_lines(csv_file_path):
    # 初始化存储最后两行的变量
    last_line = None
    second_last_line = None

    # 以二进制模式打开文件（避免解码整个文件）
    with open(csv_file_path, 'rb') as file:
        # 从文件末尾开始读取（利用seek的负偏移量）
        file.seek(-2, 2)  # 2表示从文件末尾，-2表示向前移动2字节
        # 循环向前查找换行符
        while file.read(1) != b'\n':  # 直到找到换行符
            try:
                # 向前移动两个字节（若已到文件开头则抛出异常）
                file.seek(-2, 1)  # 1表示相对当前位置
            except OSError:
                # 到达文件开头，跳出循环
                file.seek(0)
                break

        # 现在文件指针在倒数第一行（或唯一一行）的开头
        last_line = file.readline().decode().strip()
        
        # 尝试读取倒数第二行
        try:
            file.seek(-len(last_line.encode()) - 2, 1)  # 回退上一行末尾
            while file.read(1) != b'\n':
                file.seek(-2, 1)
        except OSError:
            # 没有更多行了，只有一行数据
            return [last_line]
        
        # 读取倒数第二行
        second_last_line = file.readline().decode().strip()
        return [second_last_line, last_line]

# 使用示例
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)
    
    result = get_last_two_lines(sys.argv[1])
    print("倒数第二行:", result[0] if len(result) > 1 else "N/A")
    print("最后一行:  ", result[-1])