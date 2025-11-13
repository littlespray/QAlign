#!/usr/bin/env python3
"""
Split dataset script
从train.txt和val.txt读取文件名，将源文件复制到对应的训练集和验证集目录
"""

import os
import shutil
from pathlib import Path

def split_dataset():
    # 定义路径
    base_dir = Path("IE-R1-4K")
    src_dir = base_dir / "dst"
    train_dir = base_dir / "dst_train"
    val_dir = base_dir / "dst_val"
    
    train_txt = "IE-R1-4K/train.txt"
    val_txt = "IE-R1-4K/val.txt"
    
    # 创建目标目录
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"源目录: {src_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    print("-" * 50)
    
    # 处理训练集
    if os.path.exists(train_txt):
        print(f"\n处理训练集 ({train_txt})...")
        with open(train_txt, 'r', encoding='utf-8') as f:
            train_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 使用|分隔，取第一个元素作为文件名
                filename = line.split('|')[0]
                src_file = src_dir / filename
                dst_file = train_dir / filename
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    train_count += 1
                    if train_count % 100 == 0:
                        print(f"  已复制 {train_count} 个文件...")
                else:
                    print(f"  警告: 文件不存在 - {src_file}")
            
            print(f"训练集完成: 共复制 {train_count} 个文件")
    else:
        print(f"错误: {train_txt} 不存在")
    
    # 处理验证集
    if os.path.exists(val_txt):
        print(f"\n处理验证集 ({val_txt})...")
        with open(val_txt, 'r', encoding='utf-8') as f:
            val_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 使用|分隔，取第一个元素作为文件名
                filename = line.split('|')[0]
                src_file = src_dir / filename
                dst_file = val_dir / filename
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    val_count += 1
                    if val_count % 100 == 0:
                        print(f"  已复制 {val_count} 个文件...")
                else:
                    print(f"  警告: 文件不存在 - {src_file}")
            
            print(f"验证集完成: 共复制 {val_count} 个文件")
    else:
        print(f"错误: {val_txt} 不存在")
    
    print("\n" + "=" * 50)
    print("数据集分割完成!")

if __name__ == "__main__":
    split_dataset()

