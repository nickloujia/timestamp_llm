import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import os
from config import Config

class TimestampDataset(Dataset):
    """时间戳转换数据集"""
    def __init__(self, data_file, tokenizer, max_length=Config.MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        
        # 使用简化的方法：直接格式化完整对话
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # tokenize
        tokens = self.tokenizer.encode(formatted_text, max_length=self.max_length, truncation=True)
        
        # 确保序列长度至少为2
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.eos_token_id]
        
        # 创建输入和标签
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': labels,
            'attention_mask': torch.ones_like(input_ids)
        }

class QwenSFTTrainer:
    """基于Qwen2.5-1.5B的SFT训练器"""
    def __init__(self):
        # 优先使用CUDA，其次是MPS（MacBook GPU），最后是CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型和tokenizer
        print(f"正在加载模型: {Config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=Config.TRUST_REMOTE_CODE)
        
        # 根据设备类型调整加载参数
        if self.device.type == 'mps':
            # MPS设备使用float32以提高兼容性
            torch_dtype = torch.float32
            device_map = None  # 让模型加载到CPU，然后手动移动到MPS
        else:
            torch_dtype = getattr(torch, Config.TORCH_DTYPE)
            device_map = Config.DEVICE_MAP
        
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=Config.TRUST_REMOTE_CODE
        )
        
        # 如果是MPS设备，手动将模型移动到MPS
        if self.device.type == 'mps':
            self.model = self.model.to(self.device)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 优化器（不需要单独的损失函数，使用模型内置的）
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        
        print("模型初始化完成！")
    
    def quick_evaluate(self, evaluator, num_samples=20):
        """快速评估模型性能"""
        try:
            accuracy, _ = evaluator.evaluate_model(num_samples=num_samples, verbose=False)
            return accuracy
        except Exception as e:
            print(f"快速评估出错: {e}")
            return 0.0

    def train_epoch(self, dataloader, evaluator=None, eval_interval=100):
        """训练一个epoch，只显示关键信息"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc="训练中")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 基本的数据验证
            if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
                print(f"❌ 批次 {batch_idx} input_ids 包含异常值")
                continue
            if torch.isnan(target_ids).any() or torch.isinf(target_ids).any():
                print(f"❌ 批次 {batch_idx} target_ids 包含异常值")
                continue
            
            try:
                # 前向传播 - 使用模型内置的损失计算
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=target_ids  # 模型会自动处理-100的忽略
                )
                
                loss = outputs.loss
                
                # 检查损失值是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"❌ 批次 {batch_idx} 损失值异常: {loss.item()}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 检查梯度是否包含NaN或Inf
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"❌ 批次 {batch_idx} 参数 {name} 梯度异常")
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    continue
                
                # 梯度累积
                if (batch_idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.MAX_GRAD_NORM)
                    
                    # 检查梯度范数
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"❌ 批次 {batch_idx} 梯度范数异常: {grad_norm}")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # 更新进度条
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # 定期测试
                if evaluator and (batch_idx + 1) % eval_interval == 0:
                    accuracy = self.quick_evaluate(evaluator, num_samples=20)
                    print(f"\n 批次 {batch_idx + 1} - 准确率: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"批次 {batch_idx} 训练出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return total_loss / num_batches
    
    def train(self):
        """训练模型"""
        print("开始SFT训练...")
        
        # 创建数据集和数据加载器
        train_dataset = TimestampDataset(Config.TRAIN_FILE, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        print(f"训练数据: {len(train_dataset)} 条")
        print(f"批次大小: {Config.BATCH_SIZE}")
        print(f"训练轮数: {Config.NUM_EPOCHS}")
        
        # 训练循环
        for epoch in range(Config.NUM_EPOCHS):
            print(f"\n=== Epoch {epoch + 1}/{Config.NUM_EPOCHS} ===")
            
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % 1 == 0:
                self.save_model(f"{Config.SAVE_PATH}_epoch_{epoch + 1}")
        
        # 保存最终模型
        self.save_model(Config.SAVE_PATH)
        print(f"训练完成！模型已保存到: {Config.SAVE_PATH}")
    
    def _collate_fn(self, batch):
        """数据批处理函数"""
        # 找到最大长度
        max_len = max(len(item['input_ids']) for item in batch)
        
        # 填充到相同长度
        input_ids = []
        target_ids = []
        attention_mask = []
        
        for item in batch:
            # 填充input_ids
            pad_len = max_len - len(item['input_ids'])
            input_ids.append(torch.cat([item['input_ids'], torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)]))
            # 填充target_ids - 使用-100忽略padding部分的损失
            target_ids.append(torch.cat([item['target_ids'], torch.full((pad_len,), -100, dtype=torch.long)]))
            # 创建attention_mask - padding部分为0
            mask = torch.cat([torch.ones(len(item['input_ids']), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            attention_mask.append(mask)
            
        return {
            'input_ids': torch.stack(input_ids),
            'target_ids': torch.stack(target_ids),
            'attention_mask': torch.stack(attention_mask)
        }
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        
        # 保存模型和tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        print(f"模型已保存到: {path}")

class QwenEvaluator:
    """基于Qwen模型的评估器"""
    def __init__(self, model_path=Config.SAVE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=Config.TRUST_REMOTE_CODE)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, Config.TORCH_DTYPE),
            device_map=Config.DEVICE_MAP,
            trust_remote_code=Config.TRUST_REMOTE_CODE
        )
        
        print("评估器初始化完成！")
    
    def extract_timestamp_from_response(self, response):
        """从模型回答中提取时间戳"""
        match = re.search(Config.BEIJING_TIME_PATTERN, response)
        if match:
            return match.group(1)
        return None
    
    def generate_response(self, prompt):
        """生成回答"""
        # 构建对话格式
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 使用tokenizer格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 生成回答
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_k=Config.TOP_K,
                top_p=Config.TOP_P,
                do_sample=Config.DO_SAMPLE,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回答
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def evaluate_model(self, num_samples=50, verbose=True, show_examples=True):
        """评估模型，支持静默模式和样本展示"""
        if verbose:
            print(f"开始评估模型，使用 {num_samples} 个测试样本...")
        
        # 加载测试数据
        test_data = []
        with open(Config.TEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        test_data = test_data[:num_samples]
        
        correct_count = 0
        total_count = len(test_data)
        results = []
        
        # 显示前几个样本的详细信息
        if show_examples and verbose:
            print(f"\n{'='*60}")
            print("📊 样本分析 (前5个样本)")
            print(f"{'='*60}")
        
        for i, data in enumerate(tqdm(test_data, desc="评估进度")):
            user_content = data['messages'][0]['content']
            expected_response = data['messages'][1]['content']
            
            # 提取时间戳
            timestamp_match = re.search(Config.TIMESTAMP_PATTERN, user_content)
            if not timestamp_match:
                if show_examples and i < 5:
                    print(f"❌ 样本 {i+1}: 无法从用户输入中提取时间戳")
                    print(f"   用户输入: {user_content}")
                continue
            
            timestamp = int(timestamp_match.group(1))
            
            try:
                # 生成回答
                model_response = self.generate_response(user_content)
                
                # 提取模型回答中的时间
                extracted_time = self.extract_timestamp_from_response(model_response)
                
                # 计算正确答案
                expected_time_match = re.search(Config.BEIJING_TIME_PATTERN, expected_response)
                expected_time = expected_time_match.group(1) if expected_time_match else None
                
                # 判断是否正确
                is_correct = extracted_time == expected_time
                if is_correct:
                    correct_count += 1
                
                results.append({
                    'timestamp': timestamp,
                    'expected': expected_time,
                    'predicted': extracted_time,
                    'model_response': model_response,
                    'is_correct': is_correct
                })
                
                # 显示前5个样本的详细信息
                if show_examples and i < 5 and verbose:
                    status = "✅" if is_correct else "❌"
                    print(f"\n{status} 样本 {i+1}:")
                    print(f"   时间戳: {timestamp}")
                    print(f"   用户问题: {user_content}")
                    print(f"   期望答案: {expected_time}")
                    print(f"   模型回答: {model_response}")
                    print(f"   提取结果: {extracted_time}")
                    print(f"   匹配结果: {'正确' if is_correct else '错误'}")
                
                # 每10个样本显示一次进度
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_count / (i + 1) * 100
                    if verbose:
                        print(f"\n📊 进度更新: {current_accuracy:.2f}% ({correct_count}/{i+1})")
                
            except Exception as e:
                if show_examples and i < 5:
                    print(f"❌ 样本 {i+1}: 处理出错 - {e}")
                    print(f"   用户输入: {user_content}")
                continue
        
        # 计算最终准确率
        final_accuracy = correct_count / total_count * 100
        
        if verbose:
            print(f"\n{'='*60}")
            print("📊 最终评估结果")
            print(f"{'='*60}")
            print(f"总样本数: {total_count}")
            print(f"正确回答: {correct_count}")
            print(f"准确率: {final_accuracy:.2f}%")
            
            # 显示一些错误样本
            error_samples = [r for r in results if not r['is_correct']]
            if error_samples and len(error_samples) > 0:
                print(f"\n❌ 错误样本分析 (显示前3个):")
                for j, error in enumerate(error_samples[:3]):
                    print(f"   错误 {j+1}:")
                    print(f"     时间戳: {error['timestamp']}")
                    print(f"     期望: {error['expected']}")
                    print(f"     预测: {error['predicted']}")
                    print(f"     模型回答: {error['model_response'][:100]}...")
        
        return final_accuracy, results

def check_data_format():
    """检查数据格式"""
    print("🔍 检查训练数据格式...")
    
    try:
        # 检查训练数据
        with open(Config.TRAIN_FILE, 'r', encoding='utf-8') as f:
            train_samples = []
            for i, line in enumerate(f):
                if i >= 5:  # 只检查前5个样本
                    break
                train_samples.append(json.loads(line.strip()))
        
        print(f"\n📊 训练数据样本 (前5个):")
        for i, sample in enumerate(train_samples):
            print(f"\n样本 {i+1}:")
            print(f"  用户: {sample['messages'][0]['content']}")
            print(f"  助手: {sample['messages'][1]['content']}")
            
            # 检查时间戳提取
            timestamp_match = re.search(Config.TIMESTAMP_PATTERN, sample['messages'][0]['content'])
            if timestamp_match:
                print(f"  ✅ 时间戳: {timestamp_match.group(1)}")
            else:
                print(f"  ❌ 无法提取时间戳")
            
            # 检查北京时间提取
            beijing_match = re.search(Config.BEIJING_TIME_PATTERN, sample['messages'][1]['content'])
            if beijing_match:
                print(f"  ✅ 北京时间: {beijing_match.group(1)}")
            else:
                print(f"  ❌ 无法提取北京时间")
        
        # 检查测试数据
        print(f"\n📊 测试数据样本 (前3个):")
        with open(Config.TEST_FILE, 'r', encoding='utf-8') as f:
            test_samples = []
            for i, line in enumerate(f):
                if i >= 3:  # 只检查前3个样本
                    break
                test_samples.append(json.loads(line.strip()))
        
        for i, sample in enumerate(test_samples):
            print(f"\n测试样本 {i+1}:")
            print(f"  用户: {sample['messages'][0]['content']}")
            print(f"  助手: {sample['messages'][1]['content']}")
            
            # 检查时间戳提取
            timestamp_match = re.search(Config.TIMESTAMP_PATTERN, sample['messages'][0]['content'])
            if timestamp_match:
                print(f"  ✅ 时间戳: {timestamp_match.group(1)}")
            else:
                print(f"  ❌ 无法提取时间戳")
            
            # 检查北京时间提取
            beijing_match = re.search(Config.BEIJING_TIME_PATTERN, sample['messages'][1]['content'])
            if beijing_match:
                print(f"  ✅ 北京时间: {beijing_match.group(1)}")
            else:
                print(f"  ❌ 无法提取北京时间")
        
        print(f"\n✅ 数据格式检查完成！")
        
    except Exception as e:
        print(f"❌ 数据格式检查出错: {e}")

def main():
    """主函数"""
    print("=== Qwen2.5-1.5B Timestamp转换SFT训练 ===")
    
    # 首先检查数据格式
    check_data_format()
    
    # 询问是否继续
    choice = input("\n数据格式检查完成，是否继续？(y/n): ").lower()
    if choice != 'y':
        return
    
    # 检查是否已有训练好的模型
    if os.path.exists(Config.SAVE_PATH):
        print(f"发现已训练的模型: {Config.SAVE_PATH}")
        choice = input("是否使用已训练的模型进行评估？(y/n): ").lower()
        
        if choice == 'y':
            # 使用已训练的模型进行评估
            evaluator = QwenEvaluator(Config.SAVE_PATH)
            accuracy, results = evaluator.evaluate_model(num_samples=20, show_examples=True)
            print(f"\n训练后模型准确率: {accuracy:.2f}%")
            return
    
    # 训练新模型
    print("开始训练新模型...")
    trainer = QwenSFTTrainer()
    
    # 开始训练
    trainer.train()
    
    # 评估训练后的模型
    print("\n开始评估训练后的模型...")
    evaluator = QwenEvaluator(Config.SAVE_PATH)
    accuracy, results = evaluator.evaluate_model(num_samples=50, show_examples=True)
    
    print(f"\n训练后模型准确率: {accuracy:.2f}%")
    
    if accuracy < 50:
        print("⚠️  模型准确率较低，建议增加训练数据或调整训练参数")
    elif accuracy < 80:
        print("⚠️  模型准确率中等，可以考虑进一步优化")
    else:
        print("✅ 模型表现良好！")

if __name__ == "__main__":
    main() 