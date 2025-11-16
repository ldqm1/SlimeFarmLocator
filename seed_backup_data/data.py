import sqlite3
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import os
import gc
from datetime import datetime
import subprocess

# ========== 参数配置 ==========
DB_PATH = "data.sqlite"
TABLE_NAME = "records"
DATA_FOLDER = "data"
CHUNK_SIZE = 10000  # 每批读取行数

def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            seed INTEGER PRIMARY KEY,
            count INTEGER
        );
    """)
    conn.commit()
    return conn

def remove_problematic_lines(csv_path):
    """
    逐行读取 CSV 文件，跳过引发 _csv.Error 的行。
    """
    temp_file = csv_path + ".tmp"
    error_lines = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as fin, \
             open(temp_file, 'w', encoding='utf-8', newline='') as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            line_num = 0
            while True:
                line_num += 1
                try:
                    row = next(reader)
                    writer.writerow(row)
                except StopIteration:
                    break
                except Exception as e:
                    if "_csv.Error" in str(type(e)):
                        error_lines.append(line_num)
                    else:
                        raise  # 非CSV错误则抛出
    except Exception as e:
        log(f"❌ 清理过程中发生未知错误: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False

    if error_lines:
        log(f"🗑 自动删除以下出错行号: {error_lines}")
        os.replace(temp_file, csv_path)
        return True
    else:
        os.remove(temp_file)
        return False

# ========== 日志输出 ==========
def log(msg):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}")

# ========== 获取已存在的 seed ==========
def get_existing_seeds(conn):
    try:
        return set(pd.read_sql(f"SELECT seed FROM {TABLE_NAME}", conn)["seed"].values)
    except:
        return set()

# ========== 单个 CSV 写入 ==========
def insert_csv_to_db(csv_path, conn, chunk_size):
    log(f"📄 正在处理文件：{csv_path}")

    # 设置最大字段大小
    try:
        csv.field_size_limit(min(2147483647, sys.maxsize))
    except OverflowError:
        pass

    error_count = 0
    success_count = 0

    try:
        csv_reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            engine='python',
            on_bad_lines='skip'
        )
    except Exception as e:
        log(f"⚠️ 初次读取失败: {e}")
        if remove_problematic_lines(csv_path):
            log("🔁 已尝试清理问题行，重新加载...")
            try:
                csv_reader = pd.read_csv(
                    csv_path,
                    chunksize=chunk_size,
                    engine='python',
                    on_bad_lines='skip'
                )
            except Exception as retry_e:
                log(f"❌ 二次加载仍失败: {retry_e}")
                return
        else:
            log("❌ 未发现明显问题行，无法自动修复。")
            return

    for i, chunk in enumerate(csv_reader):
        try:
            insert_unique_rows(chunk, conn, i, len(chunk))
            success_count += 1
        except Exception as e:
            error_count += 1
            log(f"⚠️ 第 {i + 1} 批数据处理失败，已跳过: {e}")
        finally:
            del chunk
            gc.collect()

    log(f"✅ 处理完成：成功 {success_count} 批，失败 {error_count} 批")

def insert_unique_rows(df, conn, i=0, len_chunk=0):
    try:
        df.columns = [c.lower() for c in df.columns]
        if not {'seed', 'count'}.issubset(df.columns):
            log("⚠️ CSV缺少必要列 seed 和 count，跳过此批次")
            return

        df = df[['seed', 'count']].copy()
        df.dropna(inplace=True)

        # 类型转换 + 去重
        df['seed'] = df['seed'].astype(int)
        df['count'] = df['count'].astype(int)
        df.drop_duplicates(subset=['seed'], keep='first', inplace=True)

        if df.empty:
            log("⚠️ 没有有效数据行")
            return

        # 使用 INSERT OR IGNORE 避免重复插入
        insert_sql = f"INSERT OR IGNORE INTO {TABLE_NAME} (seed, count) VALUES (?, ?)"
        data_tuples = list(df.itertuples(index=False, name=None))

        # 使用事务提高效率
        conn.execute("BEGIN")
        conn.executemany(insert_sql, data_tuples)
        conn.execute("COMMIT")

        log(f"✅ 成功插入 {len(data_tuples)} 条记录 - 处理第 {i + 1} 批数据，共 {len_chunk} 行")

    except Exception as e:
        conn.execute("ROLLBACK")
        log(f"⚠️ 数据处理过程中出错，跳过此批次: {e}")

# ========== 处理 backup/ 目录下所有 CSV ==========
def init_db_from_backup():
    conn = init_database()
    for fname in os.listdir(DATA_FOLDER):
        if fname.endswith(".csv"):
            fpath = os.path.join(DATA_FOLDER, fname)
            insert_csv_to_db(fpath, conn, CHUNK_SIZE)
    conn.close()
    log("✅ 所有 CSV 文件导入完成")

# ========== 查询最大 count ==========
def get_max_count_info():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(f"""
        SELECT seed, count FROM {TABLE_NAME}
        ORDER BY count DESC LIMIT 1;
    """).fetchone()
    conn.close()
    log(f"📊 最大 count: {row}")
    return row

# ========== 查询数据区间 ==========
def get_seed_range():
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(f"SELECT MIN(seed), MAX(seed) FROM {TABLE_NAME};").fetchone()
    conn.close()
    min_seed, max_seed = row[0], row[1]
    log(f"📊 数据区间: {min_seed} - {max_seed}")
    print()
    return min_seed, max_seed

def find_gap_seeds(start_seed=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(f"SELECT seed FROM {TABLE_NAME} WHERE seed > ? ORDER BY seed", (start_seed,))
    
    gaps_file = "gaps.txt"
    large_gaps = []
    is_gap = False
    
    # 缓冲区大小，控制内存使用
    BUFFER_SIZE = CHUNK_SIZE
    gap_buffer = []
    
    def flush_buffer(buffer, file_handle):
        """将缓冲区内容写入文件"""
        if buffer:
            file_handle.write('\n'.join(map(str, buffer)) + '\n')
            buffer.clear()
    
    # 打开文件准备写入小间隙
    with open(gaps_file, "w") as f:
        # 逐行读取种子，避免一次性加载到内存
        prev_seed = None
        for row in cursor:
            current_seed = row[0]
            if prev_seed is not None:
                gap = current_seed - prev_seed
                if gap > 1:
                    is_gap = True
                    start_gap = prev_seed + 1
                    end_gap = current_seed - 1
                    gap_count = end_gap - start_gap + 1
                    
                    if gap_count < 50000:
                        # 对于小间隙，使用range生成并缓冲写入
                        gap_seeds = list(range(start_gap, end_gap + 1))
                        if len(gap_buffer) + len(gap_seeds) > BUFFER_SIZE:
                            flush_buffer(gap_buffer, f)
                        gap_buffer.extend(gap_seeds)
                        # 如果缓冲区满了，立即写入
                        if len(gap_buffer) >= BUFFER_SIZE:
                            flush_buffer(gap_buffer, f)
                    else:
                        # 输出大间隙命令
                        print()
                        print(f"slime_finder --start-seed {start_gap} --threads 1 --iterations {gap_count} --output results.csv")
                        large_gaps.append(f"{start_gap}～{end_gap}")
            prev_seed = current_seed
        
        # 写入剩余的缓冲区内容
        flush_buffer(gap_buffer, f)
    
    conn.close()
    
    # 处理小间隙
    if is_gap and os.path.exists(gaps_file) and os.path.getsize(gaps_file) > 0:
        # 计算总间隙种子数（可选，如果需要精确统计）
        total_gaps = sum(1 for line in open(gaps_file))
        log(f"▶️ 开始调用 slime_finder 进行补全，共有 {total_gaps} 个间断种子")
        subprocess.run(["slime_finder", "--seed-file", gaps_file, "--threads", "15", "--output", "results.csv"])
        log("✅ slime_finder 运行完成")
        data_write()
        os.remove(gaps_file)
        log("✅ 数据补全完成")
    elif is_gap:
        log("✅ 数据连续（无小间隙需要处理）")
    else:
        log("✅ 数据连续")

def get_record_count():
    """
    查询并输出 SQLite 数据库中 records 表的总行数。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
    count = cursor.fetchone()[0]
    conn.close()
    log(f"📦 数据库当前共 {count} 条记录")
    return count

# ========== 获取所有不同count的数量 ==========
def get_distinct_counts():
    """
    获取数据库中所有不同的count值及其出现次数
    
    返回:
        dict: 键为count值，值为对应的出现次数
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        # 查询不同count值的数量
        df = pd.read_sql(f"""
            SELECT count, COUNT(*) as frequency 
            FROM {TABLE_NAME}
            GROUP BY count
            ORDER BY count DESC;
        """, conn)
        log("📊 不同count值的统计信息:")
        print(df.to_string(index=False))
        # 转换为字典：{count: frequency}
        return dict(zip(df['count'], df['frequency']))
    finally:
        conn.close()

# ========== 打印前三大count的所有数据 ==========
def print_top3_counts_data(top_count):
    """
    查询并打印前三大count值的所有种子数据
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        # 首先获取前三大不同的count值
        top_counts = conn.execute(f"""
            SELECT DISTINCT count FROM {TABLE_NAME}
            ORDER BY count DESC
            LIMIT {top_count};
        """).fetchall()
        
        if not top_counts:
            log("ℹ️ 数据库中没有数据")
            return
            
        top_counts = [count[0] for count in top_counts]
        log(f"🔝 前三大count值: {', '.join(map(str, top_counts))}")
        
        # 查询这些count值的所有数据
        for count in top_counts:
            log(f"\n📋 count = {count} 的所有记录:")
            df = pd.read_sql(f"""
                SELECT seed, count 
                FROM {TABLE_NAME}
                WHERE count = {count}
                ORDER BY seed;
            """, conn)
            
            if df.empty:
                print(f"没有count={count}的记录")
            else:
                print(f"共{len(df)}条记录:")
                print(df.to_string(index=False))
                
    finally:
        conn.close()


def data_write():
    filepath="results.csv"
    if os.path.exists(filepath):
        conn = init_database()
        insert_csv_to_db(filepath, conn, CHUNK_SIZE)
        conn.close()
        os.remove(filepath)
        print("文件删除成功")
    print("数据写入完成")
# ========== 主函数入口 ==========
def main():
    log("🚀 程序开始")
    # init_db_from_backup()

    data_write()
    get_seed_range()
    get_record_count()  
    get_max_count_info()
    find_gap_seeds(0)
    print(get_distinct_counts())
    print_top3_counts_data(2)

    # print("slime_finder --seed-file gaps.txt --threads 15 --output results.csv")
    log("🎉 全部处理完成")

if __name__ == "__main__":
    main()