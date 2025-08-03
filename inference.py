import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
import time
from datetime import datetime
import os

class TimestampInference:
    """时间戳转换推理器"""
    
    def __init__(self, model_path=Config.SAVE_PATH):
        """初始化推理器
        
        Args:
            model_path: 训练好的模型路径
        """
        self.model_path = model_path
        # 优先使用CUDA，其次是MPS（MacBook GPU），最后是CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型和tokenizer
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=Config.TRUST_REMOTE_CODE,
                pad_token=Config.PAD_TOKEN
            )
            
            # 确保pad_token设置正确
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 加载模型
            torch_dtype = torch.float32 if Config.TORCH_DTYPE == "float32" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                device_map=Config.DEVICE_MAP,
                trust_remote_code=Config.TRUST_REMOTE_CODE
            )
            
            self.model.eval()  # 设置为评估模式
            print("✅ 模型加载成功！")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请确保模型路径正确，并且已经完成训练")
            raise
    
    def extract_timestamp_from_response(self, response):
        """从模型回答中提取时间戳
        
        Args:
            response: 模型生成的回答
            
        Returns:
            str: 提取的时间戳，格式为 "YYYY年MM月DD日 HH时MM分SS秒"
        """
        pattern = Config.BEIJING_TIME_PATTERN
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        return None
    
    def generate_response(self, prompt):
        """生成模型回答
        
        Args:
            prompt: 用户输入的问题
            
        Returns:
            str: 模型生成的回答
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # 使用chat template格式化输入
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 如果chat template失败，使用简单格式
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # tokenize输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_k=Config.TOP_K,
                top_p=Config.TOP_P,
                do_sample=Config.DO_SAMPLE,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def load_test_data(self, test_file=Config.TEST_FILE):
        """加载测试数据
        
        Args:
            test_file: 测试数据文件路径
            
        Returns:
            list: 测试数据列表
        """
        test_data = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    test_data.append(item)
            
            print(f"✅ 成功加载 {len(test_data)} 条测试数据")
            return test_data
            
        except FileNotFoundError:
            print(f"❌ 测试文件不存在: {test_file}")
            return []
        except Exception as e:
            print(f"❌ 加载测试数据失败: {e}")
            return []
    
    def extract_expected_answer(self, messages):
        """从测试数据中提取期望答案
        
        Args:
            messages: 对话消息列表
            
        Returns:
            str: 期望的时间戳答案
        """
        # 假设assistant的回答包含正确的时间戳
        for message in messages:
            if message['role'] == 'assistant':
                return self.extract_timestamp_from_response(message['content'])
        return None
    
    def evaluate_single_sample(self, sample, verbose=False):
        """评估单个样本
        
        Args:
            sample: 单个测试样本
            verbose: 是否显示详细信息
            
        Returns:
            dict: 评估结果
        """
        messages = sample['messages']
        user_prompt = messages[0]['content']  # 用户问题
        expected_answer = self.extract_expected_answer(messages)  # 期望答案
        
        try:
            # 生成模型回答
            start_time = time.time()
            model_response = self.generate_response(user_prompt)
            inference_time = time.time() - start_time
            
            # 提取模型预测的时间戳
            predicted_answer = self.extract_timestamp_from_response(model_response)
            
            # 判断是否正确
            is_correct = predicted_answer == expected_answer
            
            result = {
                'prompt': user_prompt,
                'expected': expected_answer,
                'predicted': predicted_answer,
                'model_response': model_response,
                'is_correct': is_correct,
                'inference_time': inference_time
            }
            
            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"{status} 问题: {user_prompt[:50]}...")
                print(f"   期望: {expected_answer}")
                print(f"   预测: {predicted_answer}")
                print(f"   耗时: {inference_time:.3f}秒")
                if not is_correct:
                    print(f"   完整回答: {model_response}")
                print()
            
            return result
            
        except Exception as e:
            print(f"❌ 评估样本时出错: {e}")
            return {
                'prompt': user_prompt,
                'expected': expected_answer,
                'predicted': None,
                'model_response': f"Error: {e}",
                'is_correct': False,
                'inference_time': 0
            }
    
    def evaluate_test_set(self, test_file=Config.TEST_FILE, max_samples=None, verbose=True):
        """评估整个测试集
        
        Args:
            test_file: 测试数据文件路径
            max_samples: 最大测试样本数，None表示测试所有样本
            verbose: 是否显示详细信息
            
        Returns:
            dict: 评估结果统计
        """
        print("="*60)
        print("🧪 开始测试集评估")
        print("="*60)
        
        # 加载测试数据
        test_data = self.load_test_data(test_file)
        if not test_data:
            return None
        
        # 限制测试样本数
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"📊 限制测试样本数: {max_samples}")
        
        print(f"📊 总测试样本数: {len(test_data)}")
        print()
        
        # 评估所有样本
        results = []
        correct_count = 0
        total_inference_time = 0
        
        for i, sample in enumerate(test_data):
            if verbose:
                print(f"[{i+1}/{len(test_data)}]", end=" ")
            
            result = self.evaluate_single_sample(sample, verbose=verbose)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            total_inference_time += result['inference_time']
        
        # 计算统计信息
        accuracy = correct_count / len(test_data) * 100
        avg_inference_time = total_inference_time / len(test_data)
        
        # 打印结果统计
        print("="*60)
        print("📊 测试结果统计")
        print("="*60)
        print(f"✅ 正确样本数: {correct_count}")
        print(f"❌ 错误样本数: {len(test_data) - correct_count}")
        print(f"📈 准确率: {accuracy:.2f}% ({correct_count}/{len(test_data)})")
        print(f"⏱️  平均推理时间: {avg_inference_time:.3f}秒/样本")
        print(f"⏱️  总推理时间: {total_inference_time:.2f}秒")
        print("="*60)
        
        # 保存详细结果
        self.save_evaluation_results(results, accuracy)
        
        return {
            'total_samples': len(test_data),
            'correct_samples': correct_count,
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_inference_time,
            'results': results
        }
    
    def save_evaluation_results(self, results, accuracy):
        """保存评估结果到文件
        
        Args:
            results: 详细评估结果列表
            accuracy: 总体准确率
        """
        try:
            # 创建结果目录
            results_dir = "./evaluation_results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存详细结果到JSON文件
            results_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'model_path': self.model_path,
                    'accuracy': accuracy,
                    'total_samples': len(results),
                    'correct_samples': sum(1 for r in results if r['is_correct']),
                    'results': results
                }, f, ensure_ascii=False, indent=2)
            
            # 保存错误样本分析
            error_samples = [r for r in results if not r['is_correct']]
            if error_samples:
                error_file = os.path.join(results_dir, f"error_analysis_{timestamp}.txt")
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"错误样本分析 - {timestamp}\n")
                    f.write(f"总样本数: {len(results)}\n")
                    f.write(f"错误样本数: {len(error_samples)}\n")
                    f.write(f"准确率: {accuracy:.2f}%\n")
                    f.write("="*60 + "\n\n")
                    
                    for i, error in enumerate(error_samples):
                        f.write(f"错误样本 {i+1}:\n")
                        f.write(f"问题: {error['prompt']}\n")
                        f.write(f"期望答案: {error['expected']}\n")
                        f.write(f"模型预测: {error['predicted']}\n")
                        f.write(f"完整回答: {error['model_response']}\n")
                        f.write("-" * 40 + "\n\n")
            
            print(f"📋 评估结果已保存:")
            print(f"   详细结果: {results_file}")
            if error_samples:
                print(f"   错误分析: {error_file}")
            
        except Exception as e:
            print(f"⚠️ 保存评估结果时出错: {e}")
    
    def interactive_test(self):
        """交互式测试模式"""
        print("="*60)
        print("🎯 交互式测试模式")
        print("输入 'quit' 或 'exit' 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n请输入时间戳转换问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 退出交互式测试模式")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 模型思考中...")
                start_time = time.time()
                response = self.generate_response(user_input)
                inference_time = time.time() - start_time
                
                print(f"📝 模型回答: {response}")
                
                # 尝试提取时间戳
                timestamp = self.extract_timestamp_from_response(response)
                if timestamp:
                    print(f"🕐 提取的时间戳: {timestamp}")
                else:
                    print("⚠️ 未能从回答中提取到标准格式的时间戳")
                
                print(f"⏱️ 推理时间: {inference_time:.3f}秒")
                
            except KeyboardInterrupt:
                print("\n👋 退出交互式测试模式")
                break
            except Exception as e:
                print(f"❌ 处理输入时出错: {e}")

def main():
    """主函数"""
    print("🚀 时间戳转换模型推理测试")
    
    # 检查模型是否存在
    if not os.path.exists(Config.SAVE_PATH):
        print(f"❌ 模型路径不存在: {Config.SAVE_PATH}")
        print("请先运行训练脚本生成模型")
        return
    
    # 初始化推理器
    try:
        inference = TimestampInference()
    except Exception as e:
        print(f"❌ 初始化推理器失败: {e}")
        return
    
    # 选择测试模式
    print("\n请选择测试模式:")
    print("1. 测试集评估 (使用 data/test.json)")
    print("2. 交互式测试")
    print("3. 两种模式都运行")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        # 测试集评估
        max_samples = input("请输入最大测试样本数 (直接回车表示测试所有样本): ").strip()
        max_samples = int(max_samples) if max_samples.isdigit() else None
        
        evaluation_results = inference.evaluate_test_set(max_samples=max_samples)
        
        if evaluation_results:
            print(f"\n🎉 测试完成！最终准确率: {evaluation_results['accuracy']:.2f}%")
    
    if choice in ['2', '3']:
        # 交互式测试
        inference.interactive_test()

if __name__ == "__main__":
    main()