import json
import os
from collections import OrderedDict

def process_json_files(folder_path, output_file=None):
    """
    处理文件夹中的所有JSON文件，合并并排序
    
    Args:
        folder_path: 包含JSON文件的文件夹路径
        output_file: 输出文件路径（可选）
    
    Returns:
        合并后的有序字典
    """
    # 存储所有数据的字典
    merged_data = {}
    
    # 遍历文件夹中的所有JSON文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 合并数据
                for key, value in data.items():
                    if key in merged_data:
                        # 如果键已存在，合并数组并去重
                        merged_data[key].extend(value)
                        # 去重并保持int64类型
                        merged_data[key] = list(set(merged_data[key]))
                    else:
                        merged_data[key] = value
                        
            except json.JSONDecodeError:
                print(f"警告: {filename} 不是有效的JSON文件，已跳过")
            except Exception as e:
                print(f"警告: 处理 {filename} 时出错: {e}")
    
    # 对数组内容进行排序（从小到大）
    for key in merged_data:
        merged_data[key].sort()
    
    # 按键从大到小排序
    sorted_keys = sorted(merged_data.keys(), key=int, reverse=True)
    ordered_data = OrderedDict()
    
    for key in sorted_keys:
        ordered_data[key] = merged_data[key]
    
    # 如果指定了输出文件，则保存
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    
    return ordered_data

# 使用示例
if __name__ == "__main__":
    # 请修改为你的文件夹路径
    folder_path = "./used"  # 替换为你的JSON文件夹路径
    output_file = "seed_0-380b.json"  # 输出文件名
    
    # 处理文件
    result = process_json_files(folder_path, output_file)
    
    # 打印结果预览
    print("合并结果预览:")
    for i, (key, value) in enumerate(result.items()):
        if i < 5:  # 只显示前5个
            print(f"键 {key}: {value}")
        else:
            print("...")
            break
    
    print(f"\n总共处理了 {len(result)} 个不同的键")