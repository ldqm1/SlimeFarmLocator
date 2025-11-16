import sqlite3
import os
import json
import time
import sys

# === 配置参数 ===
BACKUP_DIR = "backup"
RESTORE_DB_PATH = "restored.sqlite"
TABLE_NAME = "records"
BITS_PER_COUNT = 5
CHUNK_SIZE = 100_000
BLOCK_SIZE = CHUNK_SIZE * BITS_PER_COUNT // 8  # 62500 bytes
META_PATH = os.path.join(BACKUP_DIR, "metadata.json")

# === 加载元数据 ===
def load_metadata():
    with open(META_PATH, "r") as f:
        return json.load(f)

# === 初始化还原数据库 ===
def init_restore_db():
    conn = sqlite3.connect(RESTORE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            seed INTEGER PRIMARY KEY,
            count INTEGER
        );
    """)
    conn.commit()
    return conn

# === 解码单个块（62500字节 → 100000个count）===
def decode_block(data, offset):
    bitstream = int.from_bytes(data, byteorder='big')
    counts = []
    total_bits = len(data) * 8
    for i in range(CHUNK_SIZE):
        shift = total_bits - (i + 1) * BITS_PER_COUNT
        val = (bitstream >> shift) & ((1 << BITS_PER_COUNT) - 1)
        counts.append(val + offset)
    return counts

# === 还原指定卷 ===
def restore_volume(volume_number):
    meta = load_metadata()
    offset = meta["offset"]
    volume_path = os.path.join(BACKUP_DIR, f"archive_{volume_number:03d}.bin")

    if not os.path.exists(volume_path):
        print(f"❌ 卷文件不存在: {volume_path}")
        return

    conn = init_restore_db()
    cursor = conn.cursor()

    seed = volume_number * 1_000_000_000
    block_count = 0
    start_time = time.time()

    with open(volume_path, "rb") as f:
        while True:
            block = f.read(BLOCK_SIZE)
            if not block:
                break
            if len(block) != BLOCK_SIZE:
                print(f"⚠️ 非完整块（{len(block)} 字节），中止处理")
                break

            counts = decode_block(block, offset)
            records = [(seed + i, count) for i, count in enumerate(counts)]
            cursor.executemany(f"INSERT INTO {TABLE_NAME} (seed, count) VALUES (?, ?)", records)
            seed += CHUNK_SIZE
            block_count += 1

            # === 进度显示 ===
            elapsed = time.time() - start_time
            percent = (block_count * CHUNK_SIZE) / 1_000_000_000 * 100
            avg_time = elapsed / block_count
            remaining = (1_000_000_000 - block_count * CHUNK_SIZE) // CHUNK_SIZE
            eta = remaining * avg_time

            print(f"📊 卷 {volume_number:03d} 进度: {seed} / {volume_number + 1}e9 ({percent:.2f}%) | ETA: {eta:.1f}s")

    conn.commit()
    conn.close()
    print(f"✅ 卷 {volume_number:03d} 还原完成，共写入 {seed - volume_number * 1_000_000_000} 条记录")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python data_unzip.py <卷号>")
    else:
        restore_volume(int(sys.argv[1]))