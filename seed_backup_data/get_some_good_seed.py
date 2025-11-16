import json
import sqlite3
from datetime import datetime
import pandas as pd # pyright: ignore[reportMissingModuleSource]
DB_PATH = "data.sqlite"
TABLE_NAME = "records"
DATA_FOLDER = "data"
CHUNK_SIZE = 10000  # 每批读取行数


def log(msg):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}")


def export_counts_with_seeds(min_count_threshold, output_json_path="output.json"):
    """
    查询数据库中 count > min_count_threshold 的所有记录，
    并按 count 分组，将 seed 收集为列表，导出为 JSON 文件。

    参数：
        min_count_threshold (int): 最小 count 阈值
        output_json_path (str): 输出 JSON 文件路径，默认是 "output.json"
    """
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 查询符合条件的数据
        df = pd.read_sql(f"""
            SELECT seed, count FROM {TABLE_NAME}
            WHERE count > ?
            ORDER BY count DESC, seed ASC
        """, conn, params=(min_count_threshold,))

        if df.empty:
            log(f"ℹ️ 没有找到 count > {min_count_threshold} 的记录")
            result = {}
        else:
            # 按 count 分组，聚合 seeds 为列表
            grouped = df.groupby('count')['seed'].apply(list).to_dict()
            result = grouped
            log(f"📦 已提取 {len(grouped)} 个不同的 count 值，写入 JSON 中...")

        # 写入 JSON 文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        log(f"✅ 结果已保存至 '{output_json_path}'")
    
    except Exception as e:
        log(f"❌ 导出失败: {e}")
    finally:
        conn.close()


def get_count_range():
    """
    获取数据库中 count 字段的最小值和最大值（单位亿）
    """
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 获取最小值和最大值
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN(seed), MAX(seed) FROM {TABLE_NAME}")
        min_count, max_count = cursor.fetchone()
        print(min_count,max_count)
        # 转换为亿为单位（整数）
        min_count_billion = min_count // 100000000 if min_count else 0
        max_count_billion = (max_count+1) // 100000000 if max_count else 0
        
        return min_count_billion, max_count_billion
    
    except Exception as e:
        log(f"❌ 获取count范围失败: {e}")
        return 0, 0
    finally:
        conn.close()


if __name__ == "__main__":
    # 设置最小 count 阈值
    MIN_COUNT_THRESHOLD = 55
    
    # 获取数据库中 count 的范围（单位亿）
    min_billion, max_billion = get_count_range()
    
    # 构造输出文件名：seed_{a}-{b}.json
    output_filename = f"seed_{min_billion}-{max_billion}.json"
    output_path = f"{DATA_FOLDER}/{output_filename}"
    
    export_counts_with_seeds(MIN_COUNT_THRESHOLD, output_json_path=output_path)