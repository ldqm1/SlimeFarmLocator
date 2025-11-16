import sqlite3
import sys
import os
from datetime import datetime

def verify_seed_range_chunked(original_db_path, restored_db_path, table_name, 
                            start_seed, end_seed, chunk_size=1000000, max_mismatch=10):
    """
    分块校验两个数据库在指定 seed 范围内的记录是否一致，避免内存溢出。

    参数:
        original_db_path (str): 原始数据库路径
        restored_db_path (str): 还原数据库路径
        table_name (str): 表名
        start_seed (int): 起始 seed（包含）
        end_seed (int): 结束 seed（包含）
        chunk_size (int): 每个区块的大小（默认100万）
        max_mismatch (int): 最大允许显示的不一致条数（默认10）

    返回:
        tuple: (是否一致, 发现的错误列表)
    """
    conn_orig = sqlite3.connect(original_db_path)
    conn_rest = sqlite3.connect(restored_db_path)
    
    all_errors = []
    current_chunk_start = start_seed
    processed_records = 0
    
    print(f"🔍 开始分块验证，区块大小: {chunk_size:,}")
    
    while current_chunk_start <= end_seed:
        current_chunk_end = min(current_chunk_start + chunk_size - 1, end_seed)
        
        # 获取当前区块的数据
        cursor_orig = conn_orig.execute(
            f"SELECT seed, count FROM {table_name} WHERE seed BETWEEN ? AND ? ORDER BY seed",
            (current_chunk_start, current_chunk_end)
        )
        rows_orig = cursor_orig.fetchall()
        
        cursor_rest = conn_rest.execute(
            f"SELECT seed, count FROM {table_name} WHERE seed BETWEEN ? AND ? ORDER BY seed",
            (current_chunk_start, current_chunk_end)
        )
        rows_rest = cursor_rest.fetchall()
        
        # 释放游标资源
        cursor_orig.close()
        cursor_rest.close()
        
        # 比较当前区块的数据
        chunk_errors = compare_chunks(rows_orig, rows_rest, 
                                    current_chunk_start, current_chunk_end,
                                    max_mismatch, len(all_errors))
        all_errors.extend(chunk_errors)
        
        processed_records += len(rows_orig)
        
        # 打印进度
        progress = ((current_chunk_end - start_seed + 1) / (end_seed - start_seed + 1)) * 100
        print(f"📊 进度: {progress:.1f}% - 区块 {current_chunk_start:,}-{current_chunk_end:,} "
              f"(共 {processed_records:,} 条记录)")
        
        current_chunk_start = current_chunk_end + 1
        
        # 如果有太多错误，提前终止
        if len(all_errors) >= max_mismatch * 3:  # 给一些缓冲空间
            print("⚠️ 错误过多，提前终止验证")
            break
    
    # 关闭连接
    conn_orig.close()
    conn_rest.close()
    
    # 最终统计
    is_success = len(all_errors) == 0
    
    if is_success:
        print(f"✅ 校验通过：seed 范围 {start_seed:,} – {end_seed:,} 完全一致")
        print(f"📈 总共处理了 {processed_records:,} 条记录")
    else:
        print(f"❌ 校验失败：发现 {len(all_errors)} 处不一致")
        print(f"📈 总共处理了 {processed_records:,} 条记录")
        
        if len(all_errors) > max_mismatch:
            print(f"⚠️ 超过最大显示不一致数 {max_mismatch}，仅显示前 {max_mismatch} 条错误")
    
    return is_success, all_errors

def compare_chunks(rows_orig, rows_rest, chunk_start, chunk_end, max_mismatch, existing_error_count):
    """
    比较单个区块的数据
    """
    errors = []
    
    # 检查行数差异
    if len(rows_orig) != len(rows_rest):
        error_msg = f"区块 {chunk_start:,}-{chunk_end:,}: 行数不匹配 (原库:{len(rows_orig)}, 还原库:{len(rows_rest)})"
        errors.append(error_msg)
        if existing_error_count + len(errors) <= max_mismatch:
            print(f"❌ {error_msg}")
    
    # 逐行比较
    min_len = min(len(rows_orig), len(rows_rest))
    
    for i in range(min_len):
        seed_o, count_o = rows_orig[i]
        seed_r, count_r = rows_rest[i]
        
        if seed_o != seed_r:
            error_msg = f"种子序列错位: 位置{i+1}, 期望seed={seed_o}, 实际seed={seed_r}"
            errors.append(error_msg)
            if existing_error_count + len(errors) <= max_mismatch:
                print(f"❌ {error_msg}")
            continue
            
        if count_o != count_r:
            error_msg = f"计数值不匹配: seed={seed_o:,}, 原库值={count_o}, 还原库值={count_r}"
            errors.append(error_msg)
            if existing_error_count + len(errors) <= max_mismatch:
                print(f"❌ {error_msg}")
    
    # 处理多余的行
    if len(rows_orig) > len(rows_rest):
        missing_in_rest = len(rows_orig) - len(rows_rest)
        error_msg = f"区块 {chunk_start:,}-{chunk_end:,}: 还原库缺失 {missing_in_rest} 行数据"
        errors.append(error_msg)
        if existing_error_count + len(errors) <= max_mismatch:
            print(f"❌ {error_msg}")
    elif len(rows_rest) > len(rows_orig):
        extra_in_rest = len(rows_rest) - len(rows_orig)
        error_msg = f"区块 {chunk_start:,}-{chunk_end,:}: 还原库多出 {extra_in_rest} 行数据"
        errors.append(error_msg)
        if existing_error_count + len(errors) <= max_mismatch:
            print(f"❌ {error_msg}")
    
    return errors

def verify_with_progress_monitoring(original_db_path, restored_db_path, table_name, 
                                  start_seed, end_seed, chunk_size=500000, max_mismatch=10):
    """
    带进度监控的分块验证版本（更安全的内存使用）
    """
    def get_total_records(db_path, start_seed, end_seed):
        """获取总记录数以计算准确进度"""
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE seed BETWEEN ? AND ?",
            (start_seed, end_seed)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    print("📊 正在统计总记录数...")
    try:
        total_records = get_total_records(original_db_path, start_seed, end_seed)
        print(f"📈 预计验证 {total_records:,} 条记录")
    except Exception as e:
        print(f"⚠️ 无法获取精确记录数: {e}")
        total_records = None
    
    return verify_seed_range_chunked(
        original_db_path, restored_db_path, table_name,
        start_seed, end_seed, chunk_size, max_mismatch
    )

def write_failure_log(term, errors, log_file="verification_failure.log"):
    """将验证失败信息写入日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"[{timestamp}] 第 {term} 卷验证失败\n")
        f.write(f"Seed 范围: {term * 1000000000:,} - {(term + 1) * 1000000000 - 1:,}\n")
        f.write(f"发现错误数量: {len(errors)}\n")
        f.write("-" * 40 + "\n")
        
        # 写入前20个错误详情
        for i, error in enumerate(errors[:20], 1):
            f.write(f"{i}. {error}\n")
        
        if len(errors) > 20:
            f.write(f"... 还有 {len(errors) - 20} 个错误未显示\n")
        
        f.write("="*60 + "\n")

def yanzheng(term, chunk_size=None):
    seed_start = term * 10_0000_0000
    seed_end = seed_start + 9_9999_9999
    
    # 自动选择合适的区块大小
    if chunk_size is None:
        # 根据系统内存估计合适的区块大小
        available_gb = psutil.virtual_memory().available / (1024**3) if 'psutil' in globals() else 4
        auto_chunk_size = max(100000, min(2000000, int(available_gb * 250000)))  # 经验公式
        print(f"🤖 自动选择区块大小: {auto_chunk_size:,}")
        used_chunk_size = auto_chunk_size
    else:
        used_chunk_size = chunk_size
    
    success, errors = verify_with_progress_monitoring(
        original_db_path="data.sqlite",
        restored_db_path="restored.sqlite",
        table_name="records",
        start_seed=seed_start,
        end_seed=seed_end,
        chunk_size=used_chunk_size
    )
    
    return success, errors

# 备用的简单版本（不需要psutil）
def yanzheng_simple(term, chunk_size=500000):
    seed_start = term * 10_0000_0000
    seed_end = seed_start + 9_9999_9999
    
    success, errors = verify_seed_range_chunked(
        original_db_path="data.sqlite",
        restored_db_path="restored.sqlite",
        table_name="records",
        start_seed=seed_start,
        end_seed=seed_end,
        chunk_size=chunk_size
    )
    
    return success, errors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python yanzheng.py <卷号> [区块大小]")
        print("示例:")
        print("  python yanzheng.py 1           # 使用默认区块大小")
        print("  python yanzheng.py 1 1000000   # 指定区块大小为100万")
        sys.exit(1)
    
    try:
        term = int(sys.argv[1])
        chunk_size = int(sys.argv[2]) if len(sys.argv) >= 3 else None
        
        print(f"🎯 开始验证第 {term} 卷数据库")
        print(f"🔍 Seed 范围: {term * 1000000000:,} - {(term + 1) * 1000000000 - 1:,}")
        
        # 尝试导入psutil进行智能内存管理
        try:
            import psutil
            success, errors = yanzheng(term, chunk_size)
        except ImportError:
            print("⚠️ 未安装psutil，使用简单模式")
            chunk_size = chunk_size if chunk_size else 800000
            success, errors = yanzheng_simple(term, chunk_size)
        
        if success:
            print("✅ 验证成功，删除临时数据库文件...")
            try:
                os.remove("restored.sqlite")
                print("✅ 临时文件已清理")
            except FileNotFoundError:
                print("⚠️ 临时文件不存在，无需清理")
        else:
            print("❌ 验证失败，正在写入日志...")
            write_failure_log(term, errors)
            print(f"📝 错误日志已写入 verification_failure.log")
            
    except ValueError:
        print("错误：卷号和区块大小必须是整数")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)