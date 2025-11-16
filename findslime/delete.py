import sqlite3

# ========== 可配置参数 ==========
DB_PATH = "data.sqlite"          # 数据库文件路径
TABLE_NAME = "records"           # 表名
SEED_MIN_THRESHOLD = 1_9999_9999    # seed最小阈值
SEED_MAX_THRESHOLD = 1000_0000_0000    # seed最大阈值
# ===============================

def delete_outside_seed_range_data():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 删除seed小于最小阈值或大于最大阈值的记录
        query = f"DELETE FROM {TABLE_NAME} WHERE seed < ? OR seed > ?"
        
        # 执行删除操作
        cursor.execute(query, (SEED_MIN_THRESHOLD, SEED_MAX_THRESHOLD))
        
        # 获取删除的行数
        deleted_rows = cursor.rowcount
        
        # 提交事务
        conn.commit()
        
        print(f"已删除 {deleted_rows} 条seed < {SEED_MIN_THRESHOLD} 或 seed > {SEED_MAX_THRESHOLD}的记录")
        print(f"保留seed在 [{SEED_MIN_THRESHOLD}, {SEED_MAX_THRESHOLD}] 区间内的记录")
        return deleted_rows
        
    except sqlite3.Error as e:
        # 发生错误时回滚
        if conn:
            conn.rollback()
        print(f"数据库操作错误: {e}")
        return 0
        
    finally:
        # 确保连接总是关闭
        if conn:
            conn.close()

# 调用函数执行删除操作
delete_outside_seed_range_data()