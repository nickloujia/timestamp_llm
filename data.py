import json
import random
from datetime import datetime, timedelta
import pytz

def generate_timestamp_data(num_samples):
    """生成timestamp到北京时间转换的数据"""
    data = []
    
    # 定义不同的时间戳范围
    # Unix timestamp范围：从2020年到2025年
    start_timestamp = 1577836800  # 2020-01-01 00:00:00 UTC
    end_timestamp = 1735689600    # 2025-01-01 00:00:00 UTC
    
    # 北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    for i in range(num_samples):
        # 随机生成timestamp
        timestamp = random.randint(start_timestamp, end_timestamp)
        
        # 转换为UTC时间
        utc_time = datetime.utcfromtimestamp(timestamp)
        utc_time = pytz.utc.localize(utc_time)
        
        # 转换为北京时间
        beijing_time = utc_time.astimezone(beijing_tz)
        
        # 格式化时间
        beijing_str = beijing_time.strftime('%Y年%m月%d日 %H时%M分%S秒')
        
        # 生成不同的提示格式
        prompt_formats = [
            f"请将时间戳 {timestamp} 转换为北京时间。",
            f"时间戳 {timestamp} 对应的北京时间是什么？",
            f"帮我转换时间戳 {timestamp} 为北京时间。",
            f"Unix时间戳 {timestamp} 转换为北京时间是多少？",
            f"时间戳 {timestamp} 在北京时间下是什么时候？"
        ]
        
        prompt = random.choice(prompt_formats)
        
        # 生成回答 - 直接返回时间，不加前缀
        response = beijing_str
        
        # 创建SFT格式的数据
        sft_data = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        }
        
        data.append(sft_data)
    
    return data

def generate_special_cases():
    """生成一些特殊情况的数据"""
    special_data = []
    
    # 特殊情况：夏令时、特殊时间点等
    special_timestamps = [
        (1640995200, "2022年01月01日 08时00分00秒"),  # 2022年元旦
        (1648771200, "2022年04月01日 08时00分00秒"),  # 2022年愚人节
        (1654041600, "2022年06月01日 08时00分00秒"),  # 2022年儿童节
        (1661990400, "2022年09月01日 08时00分00秒"),  # 2022年开学
        (1672531200, "2023年01月01日 08时00分00秒"),  # 2023年元旦
        (1680307200, "2023年04月01日 08时00分00秒"),  # 2023年愚人节
        (1693526400, "2023年09月01日 08时00分00秒"),  # 2023年开学
        (1704067200, "2024年01月01日 08时00分00秒"),  # 2024年元旦
        (1711929600, "2024年04月01日 08时00分00秒"),  # 2024年愚人节
        (1725148800, "2024年09月01日 08时00分00秒"),  # 2024年开学
    ]
    
    for timestamp, beijing_time in special_timestamps:
        prompt_formats = [
            f"请将时间戳 {timestamp} 转换为北京时间。",
            f"时间戳 {timestamp} 对应的北京时间是什么？",
            f"帮我转换时间戳 {timestamp} 为北京时间。"
        ]
        
        for prompt_format in prompt_formats:
            # 直接返回时间，不加前缀
            response = beijing_time
            
            sft_data = {
                "messages": [
                    {"role": "user", "content": prompt_format},
                    {"role": "assistant", "content": response}
                ]
            }
            special_data.append(sft_data)
    
    return special_data

def main():
    """主函数：生成训练和测试数据"""
    print("开始生成数据...")
    
    # 生成训练数据
    print("生成训练数据...")
    train_data = generate_timestamp_data(50000)
    
    # 添加特殊情况数据到训练集
    special_cases = generate_special_cases()
    train_data.extend(special_cases)
    
    # 生成测试数据
    print("生成测试数据...")
    test_data = generate_timestamp_data(1000)
    
    # 保存训练数据
    print("保存训练数据...")
    with open('data/train.json', 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存测试数据
    print("保存测试数据...")
    with open('data/test.json', 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据生成完成！")
    print(f"训练数据：{len(train_data)} 条")
    print(f"测试数据：{len(test_data)} 条")
    print(f"数据已保存到 data/ 目录")

if __name__ == "__main__":
    main()
