# Unix时间戳到北京时间转换 - LLM微调项目

本项目使用Qwen2.5-1.5B模型进行SFT（Supervised Fine-Tuning）训练，专门用于Unix时间戳到北京时间的转换任务。

## 🚀 快速开始

### 1. 生成训练数据
```bash
python data.py
```
这将生成：
- `data/train.json`: 50,030条训练数据
- `data/test.json`: 1,000条测试数据

### 2. 配置训练参数
编辑 `config.py` 文件，调整训练参数：
```python
# 主要配置项
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_EPOCHS = 2  # 推荐2轮训练
LEARNING_RATE = 2e-5  # 学习率
MAX_SAMPLES = None  # 使用全部数据
```

### 3. 开始训练
```bash
python train_sft_stable.py
```

### 4. 可视化训练过程
```bash
python visualize_training.py
```

## 📊 项目特色

### 🎯 **智能标签掩码**
- 只训练时间戳转换部分，忽略用户问题
- 避免学习无关的对话模式
- 提高训练效率和准确性

### 📈 **实时监控**
- 训练过程中实时评估准确率
- 自动生成Loss和准确率图表
- 详细的训练日志和报告

### 🔍 **数据质量保证**
- 自动数据格式检查
- 样本展示和错误分析
- 正则表达式验证

## 📁 文件结构

```
timestamp_llm/
├── data.py                 # 数据生成脚本
├── config.py              # 配置文件
├── train_sft_stable.py    # 主训练脚本（推荐）
├── train_sft.py           # 简化训练脚本
├── visualize_training.py  # 训练可视化
├── inference.py           # 推理测试
├── data/
│   ├── train.json         # 训练数据
│   └── test.json          # 测试数据
└── training_logs/         # 训练日志和图表
```

## 🎯 数据格式

### 输入格式（用户问题）：
```
请将时间戳 1674012218 转换为北京时间。
时间戳 1619791726 在北京时间下是什么时候？
帮我转换时间戳 1658153133 为北京时间。
```

### 输出格式（模型回答）：
```
2023年01月18日 11时23分38秒
2021年04月30日 22时08分46秒
2022年07月18日 22时05分33秒
```

## 📊 监控指标

### 训练过程监控：
1. **Training Loss**: 训练损失变化
2. **Learning Rate**: 学习率调度
3. **Test Accuracy**: 测试集准确率（100样本）
4. **Training Time**: 训练时间统计

### 输出文件：
- `training_log_YYYYMMDD_HHMMSS.csv`: 训练日志
- `accuracy_log_YYYYMMDD_HHMMSS.csv`: 准确率记录
- `training_progress_step_XXX.png`: 训练图表
- `training_report.txt`: 最终报告