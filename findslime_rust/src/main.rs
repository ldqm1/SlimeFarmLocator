use std::fs::{File, OpenOptions};
use std::io::{self, BufRead};
use std::sync::{Arc, Mutex};
use std::thread;

use clap::Parser;
use csv::Writer;

const DIM: usize = 1251;    // 遮罩尺寸: -625..=625
const OFFSET: i32 = 625;    // 坐标偏移量

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// 起始种子
    #[arg(short, long)]
    start_seed: Option<i64>,

    /// 线程数
    #[arg(short, long, default_value_t = 4)]
    threads: usize,

    /// 每线程迭代次数
    #[arg(short, long)]
    iterations: Option<usize>,

    /// 输入种子文件
    #[arg(short = 'f', long)]
    seed_file: Option<String>,

    /// 输出路径
    #[arg(short, long, default_value_t = String::from("results.csv"))]
    output: String,
}

/// 原版 check_slime 数学逻辑，用于预计算遮罩
fn check_slime_math(seed: i64, x: i32, z: i32) -> bool {
    let term1 = x.wrapping_mul(x)
                 .wrapping_mul(0x4C1906) as i64;
    let term2 = (x.wrapping_mul(0x5AC0DB)) as i64;
    let term3 = (z.wrapping_mul(z)) as i64 * 0x4307A7;
    let term4 = (z.wrapping_mul(0x5F24F)) as i64;
    let mixed = seed + term1 + term2 + term3 + term4 ^ 0x3AD8025F;
    random(mixed)
}

/// Java LCG 仿真
fn random(seed: i64) -> bool {
    let mut s = (seed ^ 25214903917) & 0x0000_FFFF_FFFF_FFFF;
    s = s.wrapping_mul(25214903917)
         .wrapping_add(11)
         & 0x0000_FFFF_FFFF_FFFF;
    ((s >> 17) % 10) == 0
}
/// 生成整区块的布尔遮罩: true 表示该 (x,z) 为 slime 区块
fn generate_slime_mask(seed: i64) -> Vec<u8> {
    let mut mask = vec![0u8; DIM * DIM];
    for z_idx in 0..DIM {
        let real_z = z_idx as i32 - OFFSET;
        for x_idx in 0..DIM {
            let real_x = x_idx as i32 - OFFSET;
            if check_slime_math(seed, real_x, real_z) {
                mask[z_idx * DIM + x_idx] = 1;
            }
        }
    }
    mask
}

/// 从预计算遮罩中查询 (x,z)
#[inline(always)]
fn check_slime_mask(mask: &[u8], x: i32, z: i32) -> bool {
    let dx = (x + OFFSET) as usize;
    let dz = (z + OFFSET) as usize;
    mask[dz * DIM + dx] != 0
}

fn first_count(mask: &[u8], z: i32) -> i32 {
    let start = [ 6,  4,  3, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, 3, 4, 6];
    let end   = [10, 12, 13,14,15,15,16,16,16,16,16,15,15,14,13,12,10];
    let mut count = 0;

    // 圆环区块
    for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
        let rz = z + i as i32;
        for dx in s..=e {
            let rx = -625 + dx;
            if check_slime_mask(mask, rx, rz) {
                count += 1;
            }
        }
    }

    // 中心 3×3 扣分
    for rx in -618..=-616 {
        for dz in 7..=9 {
            if check_slime_mask(mask, rx, z + dz) {
                count -= 1;
            }
        }
    }

    count
}

fn offset_count(mask: &[u8], x: i32, z: i32) -> i32 {
    let x = x + 8;
    let z = z + 8;
    let offsets = [
        (-8,-2),(-7,-4),(-6,-5),(-5,-6),(-4,-7),(-3,-7),(-2,-8),(-1,-8),(0,-8),
        (1,-8),(2,-8),(3,-7),(4,-7),(5,-6),(6,-5),(7,-4),(8,-2),
        (-1,2),(0,2),(1,2),
    ];
    let mut count = 0;
    for &(dz, dx) in &offsets {
        if check_slime_mask(mask, x + dx, z + dz) {
            count -= 1;
        }
        if check_slime_mask(mask, x - dx + 1, z + dz) {
            count += 1;
        }
    }
    count
}

fn get_max_string(mask: &[u8]) -> (i64, i32, i32, i32) {
    let mut max_count = 0;
    let mut max_x = -625;
    let mut max_z = -625;

    for z in -625..=608 {
        let mut fc = first_count(mask, z);
        for x in -625..=608 {
            if fc > max_count {
                max_count = fc;
                max_x = x;
                max_z = z;
            }
            fc += offset_count(mask, x, z);
        }
    }

    (max_count.into(), max_x, max_z, 0)
}

/// 处理单个种子：计算、写入并立即 flush
fn process_seed(seed: i64, writer: &Arc<Mutex<Writer<File>>>) -> io::Result<()> {
    let mask = generate_slime_mask(seed);
    let (count, _, _, _) = get_max_string(&mask); // 忽略 x 和 z

    let mut wtr = writer.lock().unwrap();
    wtr.write_record(&[
        seed.to_string(),
        count.to_string(),
    ])?;
    wtr.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    if (args.start_seed.is_none() || args.iterations.is_none()) && args.seed_file.is_none() {
        eprintln!("错误: 必须提供起始种子和循环次数，或种子文件");
        std::process::exit(1);
    }

    // 构建种子列表
    let seeds = if let Some(path) = &args.seed_file {
        let f = File::open(path)?;
        io::BufReader::new(f)
            .lines()
            .filter_map(Result::ok)
            .filter_map(|l| l.parse().ok())
            .collect::<Vec<i64>>()
    } else {
        let start = args.start_seed.unwrap();
        let iters = args.iterations.unwrap();
        (0..(args.threads * iters))
            .map(|i| start + i as i64)
            .collect()
    };

    let seeds = Arc::new(seeds);
    let writer = Arc::new(Mutex::new(Writer::from_writer(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&args.output)?,
    )));

    // 写表头
    {
        let mut w = writer.lock().unwrap();
        w.write_record(&["seed", "count"])?;
        w.flush()?;
    }

    // 分线程处理
    let mut handles = Vec::with_capacity(args.threads);
    for thread_id in 0..args.threads {
        let seeds = Arc::clone(&seeds);
        let writer = Arc::clone(&writer);

        let handle = thread::spawn(move || -> io::Result<()> {
            for (i, &seed) in seeds.iter().enumerate() {
                if i % args.threads != thread_id {
                    continue;
                }
                // 在这里调用 process_seed，计算完即写入并 flush
                process_seed(seed, &writer)?;
            }
            Ok(())
        });

        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap()?;
    }

    Ok(())
}