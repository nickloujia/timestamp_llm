class Config:
    """训练配置类"""
    
    # 模型配置
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    TORCH_DTYPE = "float32"  # 使用float32提高数值稳定性
    DEVICE_MAP = "auto"
    TRUST_REMOTE_CODE = True
    
    # 训练超参数
    LEARNING_RATE = 1e-5  # 学习率
    WEIGHT_DECAY = 0.01  # 权重衰减
    NUM_EPOCHS = 3  # 训练轮数（测试用较少轮数）
    PER_DEVICE_TRAIN_BATCH_SIZE = 4  # 每个设备的批次大小
    GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
    MAX_GRAD_NORM = 0.3  # 梯度裁剪
    WARMUP_STEPS = 100  # 预热步数
    LOGGING_STEPS = 25  # 日志记录步数
    SAVE_STEPS = 10000  # 模型保存步数
    SAVE_STRATEGY = "steps"  # 保存策略
    FP16 = False  # 是否使用半精度训练
    
    # 数据配置
    MAX_LENGTH = 512  # 最大序列长度
    TRAIN_FILE = "data/train.json"  # 训练数据文件
    TEST_FILE = "data/test.json"  # 测试数据文件
    MAX_SAMPLES = 20000  # 最大训练样本数（用于测试）
    
    # 模型保存配置
    SAVE_PATH = "./qwen_timestamp_model"  # 模型保存路径
    
    # 生成配置
    MAX_NEW_TOKENS = 100  # 生成的最大token数
    TEMPERATURE = 0.1  # 生成温度
    TOP_K = 50  # Top-K采样
    TOP_P = 0.9  # Top-P采样
    DO_SAMPLE = True  # 是否采样
    
    # 评估配置
    EVAL_NUM_SAMPLES = 50  # 评估样本数
    EVAL_STEPS = 500  # 评估步数间隔
    
    # 日志配置
    LOG_DIR = "./training_logs"  # 训练日志目录
    PLOT_STEPS = 100  # 绘图步数间隔
    
    # 正则表达式模式
    TIMESTAMP_PATTERN = r'时间戳 (\d+)'
    BEIJING_TIME_PATTERN = r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}时\d{1,2}分\d{1,2}秒)'
    
    # Tokenizer配置
    PAD_TOKEN = '<|endoftext|>'
    QWEN_PAD_TOKEN_ID = 151643  # Qwen模型的pad_token_id