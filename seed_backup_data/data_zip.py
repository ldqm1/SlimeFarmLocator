import sqlite3
import os
import json

# === 配置参数 ===
DB_PATH = "data.sqlite"
TABLE_NAME = "records"
BACKUP_DIR = "backup"
CHUNK_SIZE = 100_000
OFFSET = 37
BITS_PER_COUNT = 5
VOLUME_SIZE = 1_000_000_000  # 每卷最大记录数
META_PATH = os.path.join(BACKUP_DIR, "metadata.json")

# === 初始化目录 ===
os.makedirs(BACKUP_DIR, exist_ok=True)

# === 元数据处理 ===
def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            return json.load(f)
    else:
        return {
            "start_seed": 0,
            "end_seed": -1,
            "count_histogram": {},
            "offset": OFFSET,
            "current_volume": 0
        }

def save_metadata(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f)

# === 卷编号与路径 ===
def get_volume_number(seed):
    return seed // VOLUME_SIZE

def get_volume_path(volume):
    return os.path.join(BACKUP_DIR, f"archive_{volume:03d}.bin")

# === 连续性验证 ===
def is_continuous(rows, expected_start):
    for i, (seed, _) in enumerate(rows):
        if seed != expected_start + i:
            print(f"❌ 非连续 seed: 期望 {expected_start + i}, 实际 {seed}")
            return False
    return True

# === 位级压缩函数 ===
def compress_block_to_bytes(rows, offset):
    bitstream = 0
    bit_length = 0
    byte_array = bytearray()

    for seed, count in rows:
        val = count - offset
        if not (0 <= val < (1 << BITS_PER_COUNT)):
            raise ValueError(f"count {count} 超出压缩范围")
        bitstream = (bitstream << BITS_PER_COUNT) | val
        bit_length += BITS_PER_COUNT

        while bit_length >= 8:
            byte = (bitstream >> (bit_length - 8)) & 0xFF
            byte_array.append(byte)
            bit_length -= 8

    if bit_length != 0:
        print("⚠️ 非对齐位流，中止处理")
        return None

    return bytes(byte_array)

# === 主归档逻辑 ===
def archive_data():
    conn = sqlite3.connect(DB_PATH)
    meta = load_metadata()
    start = meta["end_seed"] + 1
    total_written = 0
    current_volume = get_volume_number(start)
    bin_path = get_volume_path(current_volume)
    bin_file = open(bin_path, "ab")

    try:
        while True:
            cursor = conn.execute(f"""
                SELECT seed, count FROM {TABLE_NAME}
                WHERE seed >= ? AND seed < ?
                ORDER BY seed ASC
            """, (start, start + CHUNK_SIZE))
            rows = cursor.fetchall()

            if len(rows) < CHUNK_SIZE:
                print(f"⚠️ 当前块仅有 {len(rows)} 条记录，未满 {CHUNK_SIZE}，中止处理")
                break

            if not is_continuous(rows, start):
                print("⚠️ 当前块数据不连续，中止处理")
                break

            compressed = compress_block_to_bytes(rows, OFFSET)
            if compressed is None or len(compressed) != CHUNK_SIZE * BITS_PER_COUNT // 8:
                print("❌ 压缩失败或长度不匹配，中止处理")
                break

            bin_file.write(compressed)

            for _, count in rows:
                meta["count_histogram"][str(count)] = meta["count_histogram"].get(str(count), 0) + 1

            meta["end_seed"] = rows[-1][0]
            new_volume = get_volume_number(meta["end_seed"] + 1)
            if new_volume != current_volume:
                bin_file.close()
                current_volume = new_volume
                bin_path = get_volume_path(current_volume)
                bin_file = open(bin_path, "ab")
                print(f"📁 卷切换至 archive_{current_volume:03d}.bin")

            meta["current_volume"] = current_volume
            save_metadata(meta)
            print(f"✅ 写入 seed {start} 到 {meta['end_seed']}")
            start = meta["end_seed"] + 1
            total_written += CHUNK_SIZE

    finally:
        bin_file.close()
        conn.close()

    print(f"🎉 冷归档完成，共写入 {total_written} 条记录")

if __name__ == "__main__":
    archive_data()