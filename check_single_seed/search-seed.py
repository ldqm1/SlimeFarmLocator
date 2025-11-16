import os
import subprocess
import configparser
import csv
import shutil
import time
import ast

# 配置参数
JAVA_PATH = r"E:\jdk\jdk21\bin\javaw.exe"  # Java路径
SLIMEFINDER_JAR = "slimefinder-1.3.2.jar"  # Slimefinder程序路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
TEMP_DIR = os.path.join(BASE_DIR, "temp")  # 临时文件目录
IMAGES_DIR_14500 = os.path.join(BASE_DIR, "14500_images")  # 最终图像输出目录
IMAGES_DIR_15000 = os.path.join(BASE_DIR, "15000_images")

with open('list.txt', 'r', encoding='utf-8') as file:
    content = file.read().strip()
    result_list = ast.literal_eval(content)

seeds = result_list
# 固定的搜索配置
SEARCH_CONFIG = {
    "append": "false",
    "fine-search": "false",
    "output-file": "results.csv",
    "center-pos": "0c8,0c8",
    "max-width": "1250",
    "min-width": "0",
    "max-block-size": "73984",
    "min-block-size": "14500",
    "max-chunk-size": "289",
    "min-chunk-size": "51",
}

# 固定的图像配置
IMAGE_CONFIG = {
    "input-file": "1.csv",
    "output-dir": "temp_images",
    "grid-width": "1",
    "block-width": "1",
    "draw-slime-chunks": "true",
    "draw-block-mask": "true",
    "draw-chunk-mask": "true",
    "draw-center": "true"
}

def setup_directories():
    """创建必要的目录"""
    # print(f"设置目录...")
    for dir_path in [TEMP_DIR, IMAGES_DIR_15000, IMAGES_DIR_14500]:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                # print(f"创建目录: {dir_path}")
            except PermissionError as e:
                print(f"权限错误 - 无法创建目录 {dir_path}: {e}")
                return False
            except Exception as e:
                print(f"创建目录失败 {dir_path}: {e}")
                return False
    return True

def create_temp_config_files(seed):
    """为指定种子创建临时配置文件"""
    try:
        # 创建临时目录
        seed_temp_dir = os.path.join(TEMP_DIR, f"seed_{seed}")
        # print(f"创建种子临时目录: {seed_temp_dir}")
        
        if not os.path.exists(seed_temp_dir):
            os.makedirs(seed_temp_dir)
        
        # 创建mask.properties
        mask_config = configparser.ConfigParser()
        mask_config.add_section('default')
        mask_config.set('default', 'world-seed', str(seed))
        mask_config.set('default', 'despawn-sphere', 'true')
        mask_config.set('default', 'exclusion-sphere', 'true')
        mask_config.set('default', 'y-offset', '0')
        mask_config.set('default', 'chunk-weight', '0')
        
        mask_path = os.path.join(seed_temp_dir, 'mask.properties')
        with open(mask_path, 'w') as f:
            mask_config.write(f)
        # print(f"创建配置文件: {mask_path}")
        
        # 创建search.properties
        search_config = configparser.ConfigParser()
        search_config.add_section('default')
        for key, value in SEARCH_CONFIG.items():
            search_config.set('default', key, value)
        
        search_path = os.path.join(seed_temp_dir, 'search.properties')
        with open(search_path, 'w') as f:
            search_config.write(f)
        # print(f"创建配置文件: {search_path}")
        
        # 创建image.properties
        image_config = configparser.ConfigParser()
        image_config.add_section('default')
        for key, value in IMAGE_CONFIG.items():
            image_config.set('default', key, value)
        
        image_path = os.path.join(seed_temp_dir, 'image.properties')
        with open(image_path, 'w') as f:
            image_config.write(f)
        # print(f"创建配置文件: {image_path}")
        
        return seed_temp_dir
    except Exception as e:
        print(f"创建配置文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_slimefinder_search(temp_dir):
    """运行Slimefinder搜索"""
    try:
        # 构建完整的JAR路径
        jar_path = os.path.join(BASE_DIR, SLIMEFINDER_JAR)
        
        # print(f"检查文件存在性:")
        # print(f"  Java路径: {os.path.exists(JAVA_PATH)} - {JAVA_PATH}")
        # print(f"  JAR文件: {os.path.exists(jar_path)} - {jar_path}")
        # print(f"  工作目录: {os.path.exists(temp_dir)} - {temp_dir}")
        
        if not os.path.exists(JAVA_PATH):
            print(f"错误: Java路径不存在: {JAVA_PATH}")
            return False
        
        if not os.path.exists(jar_path):
            print(f"错误: JAR文件不存在: {jar_path}")
            return False
        
        if not os.path.exists(temp_dir):
            print(f"错误: 工作目录不存在: {temp_dir}")
            return False
        
        # print(f"\n执行命令: {JAVA_PATH} -jar {jar_path} -s")
        # print(f"工作目录: {temp_dir}")
        
        # 使用完整路径执行
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        result = subprocess.run(
            [JAVA_PATH, "-jar", jar_path, "-s"],
            cwd=temp_dir,
            capture_output=True, 
            text=True,
            timeout=300,
            startupinfo=startupinfo
        )
        
        # print(f"返回码: {result.returncode}")
        # if result.stdout:
            # print(f"标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"错误输出:\n{result.stderr}")
            
        # 等待一下，确保文件写入完成
        time.sleep(1)
            
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("搜索超时")
        return False
    except Exception as e:
        print(f"运行搜索时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_max_block_size_record(temp_dir):
    """从结果CSV中找到blockSize最大的记录"""
    results_path = os.path.join(temp_dir, "results.csv")
    # print(f"\n检查结果文件: {results_path}")
    # print(f"文件是否存在: {os.path.exists(results_path)}")
    
    if not os.path.exists(results_path):
        # print("临时目录中的所有文件:")
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    # print(f"  {file} (大小: {size} 字节)")
                else:
                    print(f"  {file}/")
        except Exception as e:
            print(f"  列出目录内容失败: {e}")
        print(f"结果文件不存在: {results_path}")
        return None
    
    max_block_size = -1
    max_record = None
    
    try:
        with open(results_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            # print(f"CSV文件大小: {len(content)} 字符")
            if len(content) == 0:
                print("CSV文件为空")
                return None
                
            f.seek(0)
            reader = csv.DictReader(f, delimiter=';')
            
            row_count = 0
            for row in reader:
                row_count += 1
                if 'blockSize' not in row:
                    print("错误: 找不到 blockSize 列")
                    continue
                    
                # 解析blockSize (格式为 "数字/数字")
                block_size_str = row['blockSize'].split('/')[0]
                block_size = int(block_size_str)
                
                if block_size > max_block_size:
                    max_block_size = block_size
                    max_record = row
                    
            # print(f"总共处理了 {row_count} 行数据")
            
    except Exception as e:
        print(f"解析CSV文件出错: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if max_record:
        # print(f"找到最大记录，blockSize: {max_block_size}")
        # 保存到1.csv
        csv_path = os.path.join(temp_dir, "1.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                # 写入标题行
                writer.writerow(max_record.keys())
                # 写入数据行
                writer.writerow(max_record.values())
            # print(f"保存1.csv成功: {csv_path}")
            return max_record, max_block_size
        except Exception as e:
            print(f"保存1.csv失败: {e}")
            return None
    
    print("未找到有效记录")
    return None

def run_slimefinder_image(temp_dir):
    """运行Slimefinder图像生成"""
    try:
        # 构建完整的JAR路径
        jar_path = os.path.join(BASE_DIR, SLIMEFINDER_JAR)
        
        # 检查必要的文件是否存在
        csv_path = os.path.join(temp_dir, "1.csv")
        if not os.path.exists(csv_path):
            print(f"错误: 图像输入文件不存在: {csv_path}")
            return False
        
        # print(f"\n执行图像生成命令: {JAVA_PATH} -jar {jar_path} -i")
        # print(f"工作目录: {temp_dir}")
        
        # 使用完整路径执行
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        result = subprocess.run(
            [JAVA_PATH, "-jar", jar_path, "-i"],
            cwd=temp_dir,
            capture_output=True, 
            text=True,
            timeout=300,
            startupinfo=startupinfo
        )
        
        # print(f"图像生成返回码: {result.returncode}")
        # if result.stdout:
            # print(f"图像生成标准输出:\n{result.stdout}")
        if result.stderr:
            print(f"图像生成错误输出:\n{result.stderr}")
            
        # 等待一下，确保文件写入完成
        time.sleep(1)
            
        # 检查图像是否生成
        temp_images_dir = os.path.join(temp_dir, "temp_images")
        # print(f"检查图像目录: {temp_images_dir}")
        # print(f"图像目录是否存在: {os.path.exists(temp_images_dir)}")
        
        if os.path.exists(temp_images_dir):
            # print("图像目录中的文件:")
            try:
                files = os.listdir(temp_images_dir)
                # for file in files:
                    # print(f"  {file}")
            except Exception as e:
                print(f"  列出图像目录内容失败: {e}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        # print("图像生成超时")
        return False
    except Exception as e:
        # print(f"运行图像生成时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def rename_and_move_image(temp_dir, seed, max_record, max_block_size):
    """重命名并移动生成的图像文件"""
    try:
        # 从记录中获取坐标信息
        block_position = max_record.get('block-position', '0,0')
        try:
            coords = block_position.split(',')
            if len(coords) >= 2:
                x = ''.join(filter(lambda c: c.isdigit() or c == '-', coords[0]))
                z = ''.join(filter(lambda c: c.isdigit() or c == '-', coords[1]))
            else:
                x, z = "0", "0"
        except:
            x, z = "0", "0"
        
        # 查找生成的图像文件
        temp_images_dir = os.path.join(temp_dir, "temp_images")
        if not os.path.exists(temp_images_dir):
            # print(f"临时图像目录不存在: {temp_images_dir}")
            return False
        
        image_files = [f for f in os.listdir(temp_images_dir) if f.endswith('.png')]
        if not image_files:
            print(f"没有找到生成的图像文件")
            return False
        
        # 重命名并移动第一个图像文件
        old_image_path = os.path.join(temp_images_dir, image_files[0])
        new_image_name = f"{max_block_size}b_{seed}_{x}_{z}.png"
        if max_block_size >= 15000:
            new_image_path = os.path.join(IMAGES_DIR_15000, new_image_name)
        elif max_block_size >= 14500:
            new_image_path = os.path.join(IMAGES_DIR_14500, new_image_name)
        
        shutil.copy2(old_image_path, new_image_path)
        # print(f"已保存图像: {new_image_name}")
        return True
    except Exception as e:
        print(f"重命名并移动图像时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_single_seed(seed):
    """处理单个种子的完整流程"""
    print(f"\n{'='*60}")
    print(f"处理种子: {seed}")
    print(f"{'='*60}")
    
    try:
        # 1. 创建临时配置文件
        # print("1. 创建临时配置文件...")
        temp_dir = create_temp_config_files(seed)
        if not temp_dir:
            print("  创建配置文件失败")
            return False
        
        # 2. 运行搜索
        # print("\n2. 运行搜索...")
        if not run_slimefinder_search(temp_dir):
            print("  搜索失败")
            return False
        
        # 3. 查找最大blockSize记录
        # print("\n3. 查找最大blockSize记录...")
        result = find_max_block_size_record(temp_dir)
        if not result:
            print("  未找到有效记录")
            return False
        
        max_record, max_block_size = result
        print(f"  找到最大blockSize: {max_block_size}")
        
        if max_block_size < 14500:
            print(f"  blockSize {max_block_size} 小于14500，跳过图像生成")
            return True
        # 4. 运行图像生成
        # print("\n4. 生成图像...")
        if not run_slimefinder_image(temp_dir):
            print("  图像生成失败")
            return False
        
        # 5. 重命名并移动图像
        # print("\n5. 重命名并移动图像...")
        if not rename_and_move_image(temp_dir, seed, max_record, max_block_size):
            print("  图像重命名失败")
            return False
        
        print(f"\n种子 {seed} 处理完成")
        return True
    except Exception as e:
        print(f"  处理种子 {seed} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # print(f"脚本目录: {BASE_DIR}")
    # print(f"Java路径: {JAVA_PATH}")
    # print(f"JAR文件: {os.path.join(BASE_DIR, SLIMEFINDER_JAR)}")
    
    # 检查基本文件存在性
    jar_path = os.path.join(BASE_DIR, SLIMEFINDER_JAR)
    if not os.path.exists(jar_path):
        print(f"错误: JAR文件不存在: {jar_path}")
        print("请确保 slimefinder-1.3.2.jar 文件在以下目录中:")
        print(f"  {BASE_DIR}")
        return
    
    if not os.path.exists(JAVA_PATH):
        print(f"错误: Java路径不存在: {JAVA_PATH}")
        print("请检查 Java 安装路径是否正确")
        return
    
    # 创建必要的目录
    # print("\n设置工作目录...")
    if not setup_directories():
        print("创建目录失败，请检查权限")
        return
    
    # 从seeds_with_counts 获取种子列表（全部种子）
    # seeds = [1732575694]
    # for i in seeds_with_counts.keys():
    #     seeds.extend(seeds_with_counts[i])
    print(f"\n总共需要处理的种子数量: {len(seeds)}")
    success_count = 0
    total_count = len(seeds)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n进度: {i}/{total_count}")
        if process_single_seed(seed):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成！成功: {success_count}/{total_count}")

if __name__ == "__main__":
    main()