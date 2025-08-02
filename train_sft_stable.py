import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
import re
import os
from config import Config
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv

class TimestampDataset(Dataset):
    """时间戳转换数据集 - 稳定版本"""
    def __init__(self, data_file, tokenizer, max_length=Config.MAX_LENGTH, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                
                # 如果设置了最大样本数，达到限制后停止加载
                if max_samples and len(self.data) >= max_samples:
                    break
        
        print(f"加载了 {len(self.data)} 条训练数据")
        if max_samples:
            print(f"限制最大样本数: {max_samples}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']
        
        user_msg = messages[0]['content']
        assistant_msg = messages[1]['content']
        
        # 分别tokenize用户消息和助手消息
        try:
            # 格式化用户部分
            user_formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            user_formatted = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        # 完整的对话格式
        try:
            full_formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except:
            full_formatted = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        
        # tokenize完整对话
        full_encoding = self.tokenizer(
            full_formatted,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        # tokenize用户部分（用于确定mask边界）
        user_encoding = self.tokenizer(
            user_formatted,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = full_encoding['input_ids'].squeeze()
        user_length = len(user_encoding['input_ids'].squeeze())
        
        # 确保有足够的长度
        if len(input_ids) < 2:
            return None
        
        # 创建labels，只在assistant回答部分计算loss
        labels = input_ids.clone()
        # 将用户输入部分的labels设为-100（不计算loss）
        if user_length < len(labels):
            labels[:user_length] = -100
        
        # 进一步优化：只在包含时间戳答案的部分计算loss
        # 查找assistant回答中的时间戳部分
        assistant_text = assistant_msg
        if "answer:{" in assistant_text:
            # 如果使用answer:{}格式，重点训练这部分
            answer_start = assistant_text.find("answer:{")
            if answer_start > 0:
                # 找到answer:{}在token中的位置
                prefix_text = user_formatted + assistant_text[:answer_start]
                prefix_encoding = self.tokenizer(prefix_text, add_special_tokens=False)
                prefix_length = len(prefix_encoding['input_ids'])
                
                # 只保留answer部分的loss，其他assistant文本也mask掉
                if prefix_length < len(labels):
                    labels[user_length:prefix_length] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'user_length': user_length  # 用于调试
        }

def collate_fn(batch):
    """自定义批处理函数"""
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # 获取最大长度
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # 填充input_ids - 使用配置文件中的pad_token_id
        padded_input = torch.cat([
            item['input_ids'],
            torch.full((pad_len,), Config.QWEN_PAD_TOKEN_ID, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # 填充labels，padding部分设为-100
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        labels.append(padded_labels)
        
        # attention mask
        mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }

class LossLoggingCallback(TrainerCallback):
    """自定义回调函数，用于详细记录loss并保存到文件"""
    def __init__(self, save_dir=Config.LOG_DIR):
        self.losses = []
        self.learning_rates = []
        self.steps = []
        self.timestamps = []
        self.accuracies = []  # 新增：记录准确率
        self.accuracy_steps = []  # 新增：记录准确率对应的步数
        self.start_time = time.time()
        self.step_times = []
        
        # 创建保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化CSV文件
        self.csv_file = os.path.join(save_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'learning_rate', 'elapsed_time', 'timestamp'])
        
        # 初始化准确率CSV文件
        self.accuracy_csv_file = os.path.join(save_dir, f"accuracy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.accuracy_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'accuracy', 'timestamp'])
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            current_time = time.time()
            elapsed = current_time - self.start_time
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 记录数据
            self.losses.append(logs['loss'])
            self.learning_rates.append(logs.get('learning_rate', 0))
            self.steps.append(state.global_step)
            self.timestamps.append(timestamp)
            
            # 保存到CSV文件
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step, 
                    logs['loss'], 
                    logs.get('learning_rate', 0), 
                    elapsed, 
                    timestamp
                ])
            
            # 计算平均loss（最近10步）
            recent_losses = self.losses[-10:]
            avg_loss = sum(recent_losses) / len(recent_losses)
            
            # 打印详细信息
            print(f"\n[{timestamp}] Step {state.global_step}")
            print(f"  当前Loss: {logs['loss']:.6f}")
            print(f"  平均Loss(最近10步): {avg_loss:.6f}")
            print(f"  学习率: {logs.get('learning_rate', 'N/A')}")
            print(f"  已训练时间: {elapsed/60:.1f}分钟")
            
            # 如果有loss下降趋势，显示
            if len(self.losses) >= 20:
                recent_20 = sum(self.losses[-20:-10]) / 10
                current_10 = sum(self.losses[-10:]) / 10
                improvement = recent_20 - current_10
                if improvement > 0:
                    print(f"  ✅ Loss改善: -{improvement:.6f}")
                else:
                    print(f"  ⚠️  Loss变化: {-improvement:.6f}")
            
            # 根据配置文件设置的步数保存图表
            if state.global_step % Config.PLOT_STEPS == 0 and len(self.losses) > 10:
                self.save_loss_plot()
    
    def save_loss_plot(self):
        """保存loss变化图表，包含准确率"""
        try:
            # 根据是否有准确率数据决定子图数量
            if len(self.accuracies) > 0:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 绘制loss曲线
            ax1.plot(self.steps, self.losses, 'b-', linewidth=1, alpha=0.7, label='Training Loss')
            
            # 添加移动平均线
            if len(self.losses) >= 10:
                window_size = min(50, len(self.losses) // 4)
                moving_avg = np.convolve(self.losses, np.ones(window_size)/window_size, mode='valid')
                moving_steps = self.steps[window_size-1:]
                ax1.plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} steps)')
            
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制学习率曲线
            ax2.plot(self.steps, self.learning_rates, 'g-', linewidth=1, label='Learning Rate')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            # 绘制准确率曲线（如果有数据）
            if len(self.accuracies) > 0:
                ax3.plot(self.accuracy_steps, self.accuracies, 'purple', marker='o', linewidth=2, markersize=6, label='Test Accuracy')
                ax3.set_xlabel('Training Steps')
                ax3.set_ylabel('Accuracy (%)')
                ax3.set_title('Test Set Accuracy Over Time (100 samples from data/test.json)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(0, 100)  # 准确率范围0-100%
                
                # 在准确率点上显示数值
                for i, (step, acc) in enumerate(zip(self.accuracy_steps, self.accuracies)):
                    ax3.annotate(f'{acc:.1f}%', (step, acc), textcoords="offset points", 
                               xytext=(0,10), ha='center', fontsize=9)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(self.save_dir, f"training_progress_step_{self.steps[-1]}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  📊 训练图表已保存: {plot_file}")
            if len(self.accuracies) > 0:
                print(f"  📈 包含 {len(self.accuracies)} 个准确率数据点")
            
        except Exception as e:
            print(f"  ⚠️ 保存图表时出错: {e}")
    
    def save_accuracy_to_csv(self, step, accuracy):
        """保存准确率到CSV文件"""
        try:
            with open(self.accuracy_csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([step, accuracy, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        except Exception as e:
            print(f"  ⚠️ 保存准确率数据时出错: {e}")
    
    def save_final_report(self):
        """保存最终训练报告"""
        try:
            report_file = os.path.join(self.save_dir, "training_report.txt")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=== 训练报告 ===\n")
                f.write(f"训练开始时间: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总训练时间: {(time.time() - self.start_time)/3600:.2f} 小时\n")
                f.write(f"总训练步数: {len(self.steps)}\n")
                f.write(f"初始Loss: {self.losses[0]:.6f}\n")
                f.write(f"最终Loss: {self.losses[-1]:.6f}\n")
                f.write(f"Loss改善: {self.losses[0] - self.losses[-1]:.6f}\n")
                f.write(f"最低Loss: {min(self.losses):.6f}\n")
                f.write(f"平均Loss: {np.mean(self.losses):.6f}\n")
                f.write(f"Loss标准差: {np.std(self.losses):.6f}\n")
            
            print(f"📋 训练报告已保存: {report_file}")
            
        except Exception as e:
            print(f"⚠️ 保存训练报告时出错: {e}")

class TimestampEvaluator:
    """时间戳转换评估器"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        self.test_data = self.load_test_data()
    
    def load_test_data(self):
        """加载测试数据"""
        try:
            test_data = []
            with open(Config.TEST_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    test_data.append(item)
            print(f"✅ 加载了 {len(test_data)} 条测试数据")
            return test_data
        except Exception as e:
            print(f"⚠️ 加载测试数据失败: {e}")
            return []
    
    def extract_timestamp_from_response(self, response):
        """从模型回答中提取时间戳"""
        pattern = Config.BEIJING_TIME_PATTERN
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        return None
    
    def extract_expected_answer(self, messages):
        """从测试数据中提取期望答案"""
        for message in messages:
            if message['role'] == 'assistant':
                return self.extract_timestamp_from_response(message['content'])
        return None
    
    def generate_response(self, prompt):
        """生成模型回答"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
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
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def evaluate_test_samples(self, num_samples=100):
        """使用测试集评估模型性能"""
        if not self.test_data:
            print("❌ 没有可用的测试数据")
            return 0.0
        
        # 限制评估样本数
        eval_samples = self.test_data[:min(num_samples, len(self.test_data))]
        correct = 0
        total = len(eval_samples)
        
        print(f"\n=== 测试集评估 ({total}个样本) ===")
        
        for i, sample in enumerate(eval_samples):
            try:
                messages = sample['messages']
                user_prompt = messages[0]['content']
                expected_answer = self.extract_expected_answer(messages)
                
                # 生成模型回答
                response = self.generate_response(user_prompt)
                predicted_answer = self.extract_timestamp_from_response(response)
                
                is_correct = predicted_answer == expected_answer
                if is_correct:
                    correct += 1
                
                # 每10个样本显示一次进度
                if (i + 1) % 10 == 0 or i == total - 1:
                    print(f"  进度: {i+1}/{total}, 当前准确率: {correct/(i+1)*100:.1f}%")
                
            except Exception as e:
                print(f"❌ 样本{i+1}: 评估出错 - {e}")
        
        accuracy = correct / total * 100
        print(f"📊 最终准确率: {accuracy:.1f}% ({correct}/{total})")
        return accuracy

class EvaluationCallback(TrainerCallback):
    """评估回调函数"""
    def __init__(self, model, tokenizer, loss_callback, eval_steps=Config.EVAL_STEPS):
        self.evaluator = TimestampEvaluator(model, tokenizer)
        self.eval_steps = eval_steps
        self.last_eval_step = 0
        self.loss_callback = loss_callback  # 引用loss回调以记录准确率
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step - self.last_eval_step >= self.eval_steps:
            print(f"\n{'='*50}")
            print(f"第 {state.global_step} 步评估")
            accuracy = self.evaluator.evaluate_test_samples(num_samples=100)
            
            # 将准确率记录到loss回调中
            self.loss_callback.accuracies.append(accuracy)
            self.loss_callback.accuracy_steps.append(state.global_step)
            
            # 保存准确率到CSV文件
            self.loss_callback.save_accuracy_to_csv(state.global_step, accuracy)
            
            print(f"{'='*50}")
            self.last_eval_step = state.global_step

class StableTrainer:
    """稳定的SFT训练器"""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型和tokenizer - 使用配置文件参数
        print(f"正在加载模型: {Config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            trust_remote_code=Config.TRUST_REMOTE_CODE,
            pad_token=Config.PAD_TOKEN
        )
        
        # 确保pad_token设置正确
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型，使用配置文件中的设置
        torch_dtype = torch.float32 if Config.TORCH_DTYPE == "float32" else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=Config.DEVICE_MAP,
            trust_remote_code=Config.TRUST_REMOTE_CODE
        )
        
        # 确保模型的pad_token_id正确设置
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print("模型初始化完成！")
    
    def train(self):
        """使用Transformers Trainer进行训练"""
        print("开始SFT训练...")
        
        # 创建数据集 - 使用配置文件中的最大样本数
        train_dataset = TimestampDataset(Config.TRAIN_FILE, self.tokenizer, max_samples=Config.MAX_SAMPLES)
        
        # 训练参数 - 从配置文件读取所有参数
        training_args = TrainingArguments(
            output_dir=Config.SAVE_PATH,
            num_train_epochs=Config.NUM_EPOCHS,
            per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            warmup_steps=Config.WARMUP_STEPS,
            logging_steps=Config.LOGGING_STEPS,
            save_steps=Config.SAVE_STEPS,
            save_strategy=Config.SAVE_STRATEGY,
            fp16=Config.FP16,
            max_grad_norm=Config.MAX_GRAD_NORM,
        )
        
        # 创建回调函数
        loss_callback = LossLoggingCallback(save_dir=Config.LOG_DIR)
        eval_callback = EvaluationCallback(self.model, self.tokenizer, loss_callback, eval_steps=Config.EVAL_STEPS)
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=collate_fn,
            callbacks=[loss_callback, eval_callback],
        )
        
        # 开始训练
        print("开始训练...")
        try:
            trainer.train()
            print("训练完成！")
            
            # 保存最终的loss图表和报告
            loss_callback.save_loss_plot()
            loss_callback.save_final_report()
            
            # 保存模型
            trainer.save_model(Config.SAVE_PATH)
            self.tokenizer.save_pretrained(Config.SAVE_PATH)
            print(f"模型已保存到: {Config.SAVE_PATH}")
            
            # 显示训练总结
            self.print_training_summary(loss_callback)
            
        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
    
    def print_training_summary(self, loss_callback):
        """打印训练总结"""
        if len(loss_callback.losses) > 0:
            print("\n" + "="*60)
            print("🎉 训练总结")
            print("="*60)
            print(f"📊 总训练步数: {len(loss_callback.steps)}")
            print(f"⏱️  总训练时间: {(time.time() - loss_callback.start_time)/3600:.2f} 小时")
            print(f"📉 初始Loss: {loss_callback.losses[0]:.6f}")
            print(f"📉 最终Loss: {loss_callback.losses[-1]:.6f}")
            print(f"📈 Loss改善: {loss_callback.losses[0] - loss_callback.losses[-1]:.6f}")
            print(f"🎯 最低Loss: {min(loss_callback.losses):.6f}")
            print(f"📋 训练日志保存在: {loss_callback.save_dir}")
            print("="*60)

def test_data_loading():
    """测试数据加载并显示训练目标"""
    print("测试数据加载...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, trust_remote_code=Config.TRUST_REMOTE_CODE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集 - 使用配置文件中的最大样本数
    dataset = TimestampDataset(Config.TRAIN_FILE, tokenizer, max_samples=Config.MAX_SAMPLES)
    
    # 详细分析前几个样本的训练目标
    print("\n" + "="*60)
    print("📊 训练目标分析 (显示模型实际学习的内容)")
    print("="*60)
    
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        if item is not None:
            print(f"\n样本 {i+1}:")
            print("-" * 40)
            
            input_ids = item['input_ids']
            labels = item['labels']
            user_length = item.get('user_length', 0)
            
            # 解码完整输入
            full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"完整对话:\n{full_text}")
            
            # 显示哪些部分会被训练（labels != -100）
            training_tokens = []
            training_text_parts = []
            
            for j, (token_id, label_id) in enumerate(zip(input_ids, labels)):
                if label_id != -100:  # 这些token会被训练
                    training_tokens.append(token_id.item())
                    training_text_parts.append(tokenizer.decode([token_id], skip_special_tokens=True))
            
            if training_tokens:
                training_text = tokenizer.decode(training_tokens, skip_special_tokens=True)
                print(f"\n🎯 模型实际学习的内容:")
                print(f"'{training_text}'")
                print(f"📊 训练token数: {len(training_tokens)} / {len(input_ids)} (总数)")
                print(f"📊 训练比例: {len(training_tokens)/len(input_ids)*100:.1f}%")
            else:
                print("⚠️ 警告: 没有找到训练目标token!")
            
            # 显示被mask的部分
            masked_tokens = []
            for j, (token_id, label_id) in enumerate(zip(input_ids, labels)):
                if label_id == -100:
                    masked_tokens.append(token_id.item())
            
            if masked_tokens:
                masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=True)
                print(f"🚫 被忽略的内容 (不参与训练):")
                print(f"'{masked_text}'")
    
    print("\n" + "="*60)
    
    # 测试批处理
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    if batch is not None:
        print("批处理测试:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        
        # 统计训练token比例
        total_tokens = batch['labels'].numel()
        training_tokens = (batch['labels'] != -100).sum().item()
        print(f"  训练token比例: {training_tokens/total_tokens*100:.1f}%")
    
    print("数据加载测试完成！")

def main():
    """主函数"""
    print("=== 稳定版本 Qwen SFT训练 ===")
    print(f"配置信息:")
    print(f"  模型: {Config.MODEL_NAME}")
    print(f"  最大样本数: {Config.MAX_SAMPLES}")
    print(f"  训练轮数: {Config.NUM_EPOCHS}")
    print(f"  批次大小: {Config.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  学习率: {Config.LEARNING_RATE}")
    
    # 首先测试数据加载
    test_data_loading()
    
    # 开始训练
    trainer = StableTrainer()
    trainer.train()

if __name__ == "__main__":
    main()