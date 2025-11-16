import json
import os
import glob

def process_json_files():
    # 获取当前目录下所有json文件
    json_files = glob.glob('*.json')
    
    if not json_files:
        print("当前目录下没有找到JSON文件")
        return
    
    # 用于存储所有列表的合并结果
    combined_list = []
    
    # 遍历所有JSON文件
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # 检查数据是否为字典格式
                if isinstance(data, dict):
                    # 按照键的数字大小降序排序（键大的在前）
                    sorted_keys = sorted(data.keys(), key=lambda x: int(x), reverse=True)
                    
                    for key in sorted_keys:
                        value = data[key]
                        if isinstance(value, list):
                            combined_list.extend(value)
                            print(f"文件 {file_path} 中键 '{key}' 添加了 {len(value)} 个元素")
                        else:
                            print(f"警告: 文件 {file_path} 中键 '{key}' 的值不是列表类型")
                else:
                    print(f"警告: 文件 {file_path} 不是字典格式")
                    
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        except ValueError:
            print(f"错误: 文件 {file_path} 中的键无法转换为数字")
        except Exception as e:
            print(f"处理文件 {file_path} 时发生错误: {str(e)}")
    
    # 对合并后的列表进行排序
    if combined_list:
        try:
            # 尝试排序（适用于可比较的元素）
            # combined_list.sort()
            print(f"总共合并了 {len(combined_list)} 个元素，已成功排序")
        except TypeError:
            # 如果元素类型不一致无法排序，保持原顺序
            print(f"总共合并了 {len(combined_list)} 个元素，元素类型不一致，保持原始顺序")
        except Exception as e:
            print(f"排序时发生错误: {str(e)}，保持原始顺序")
    else:
        print("没有找到任何列表数据")
        return
    
    # 将列表写入文件
    try:
        with open('list.txt', 'w', encoding='utf-8') as output_file:
            # 直接将列表格式写入文件
            output_file.write(str(combined_list))
        print("结果已成功写入 list.txt 文件")
    except Exception as e:
        print(f"写入文件时发生错误: {str(e)}")

if __name__ == "__main__":
    process_json_files()