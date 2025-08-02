"""
训练日志可视化工具
用于分析和可视化训练过程中的loss变化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, log_file):
        """初始化可视化器
        
        Args:
            log_file: CSV格式的训练日志文件路径
        """
        self.log_file = log_file
        self.data = None
        self.load_data()
    
    def load_data(self):
        """加载训练数据"""
        try:
            self.data = pd.read_csv(self.log_file)
            print(f"✅ 成功加载训练日志: {self.log_file}")
            print(f"📊 数据点数量: {len(self.data)}")
            print(f"📈 训练步数范围: {self.data['step'].min()} - {self.data['step'].max()}")
            print(f"📉 Loss范围: {self.data['loss'].min():.6f} - {self.data['loss'].max():.6f}")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            raise
    
    def plot_loss_curve(self, save_path=None, show_moving_avg=True, window_size=50):
        """绘制loss曲线
        
        Args:
            save_path: 保存路径，如果为None则显示图表
            show_moving_avg: 是否显示移动平均线
            window_size: 移动平均窗口大小
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制原始loss曲线
        plt.plot(self.data['step'], self.data['loss'], 'b-', linewidth=1, alpha=0.6, label='Training Loss')
        
        # 绘制移动平均线
        if show_moving_avg and len(self.data) >= window_size:
            moving_avg = self.data['loss'].rolling(window=window_size, center=True).mean()
            plt.plot(self.data['step'], moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} steps)')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        min_loss = self.data['loss'].min()
        max_loss = self.data['loss'].max()
        final_loss = self.data['loss'].iloc[-1]
        plt.text(0.02, 0.98, f'Min Loss: {min_loss:.6f}\nFinal Loss: {final_loss:.6f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Loss曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_learning_rate(self, save_path=None):
        """绘制学习率变化曲线"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.data['step'], self.data['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 学习率曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comprehensive_analysis(self, save_path=None):
        """绘制综合分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Loss曲线
        axes[0, 0].plot(self.data['step'], self.data['loss'], 'b-', linewidth=1, alpha=0.7)
        if len(self.data) >= 50:
            moving_avg = self.data['loss'].rolling(window=50, center=True).mean()
            axes[0, 0].plot(self.data['step'], moving_avg, 'r-', linewidth=2)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 学习率曲线
        axes[0, 1].plot(self.data['step'], self.data['learning_rate'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 3. Loss分布直方图
        axes[1, 0].hist(self.data['loss'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(self.data['loss'].mean(), color='red', linestyle='--', label=f'Mean: {self.data["loss"].mean():.6f}')
        axes[1, 0].axvline(self.data['loss'].median(), color='green', linestyle='--', label=f'Median: {self.data["loss"].median():.6f}')
        axes[1, 0].set_xlabel('Loss Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Loss Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loss改善趋势
        if len(self.data) >= 100:
            # 计算每100步的平均loss
            step_groups = self.data.groupby(self.data['step'] // 100)['loss'].mean()
            axes[1, 1].plot(step_groups.index * 100, step_groups.values, 'o-', linewidth=2, markersize=4)
            axes[1, 1].set_xlabel('Training Steps (grouped by 100)')
            axes[1, 1].set_ylabel('Average Loss')
            axes[1, 1].set_title('Loss Improvement Trend')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor trend analysis', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Loss Improvement Trend')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 综合分析图表已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_statistics_report(self, save_path=None):
        """生成统计报告"""
        stats = {
            '总训练步数': len(self.data),
            '初始Loss': self.data['loss'].iloc[0],
            '最终Loss': self.data['loss'].iloc[-1],
            '最低Loss': self.data['loss'].min(),
            '最高Loss': self.data['loss'].max(),
            '平均Loss': self.data['loss'].mean(),
            'Loss标准差': self.data['loss'].std(),
            'Loss改善': self.data['loss'].iloc[0] - self.data['loss'].iloc[-1],
            '改善百分比': ((self.data['loss'].iloc[0] - self.data['loss'].iloc[-1]) / self.data['loss'].iloc[0]) * 100
        }
        
        report = "=== 训练统计报告 ===\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"数据文件: {self.log_file}\n\n"
        
        for key, value in stats.items():
            if isinstance(value, float):
                report += f"{key}: {value:.6f}\n"
            else:
                report += f"{key}: {value}\n"
        
        # 添加训练时间分析
        if 'elapsed_time' in self.data.columns:
            total_time = self.data['elapsed_time'].iloc[-1]
            report += f"\n总训练时间: {total_time/3600:.2f} 小时\n"
            report += f"平均每步时间: {total_time/len(self.data):.2f} 秒\n"
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📋 统计报告已保存: {save_path}")
        
        return stats

def find_latest_log_file(log_dir="./training_logs"):
    """查找最新的训练日志文件"""
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"日志目录不存在: {log_dir}")
    
    csv_files = [f for f in os.listdir(log_dir) if f.startswith('training_log_') and f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"在 {log_dir} 中未找到训练日志文件")
    
    # 按修改时间排序，返回最新的
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    latest_file = os.path.join(log_dir, csv_files[0])
    
    print(f"🔍 找到最新的日志文件: {latest_file}")
    return latest_file

def main():
    parser = argparse.ArgumentParser(description='训练日志可视化工具')
    parser.add_argument('--log_file', type=str, help='训练日志CSV文件路径')
    parser.add_argument('--log_dir', type=str, default='./training_logs', help='日志目录路径')
    parser.add_argument('--output_dir', type=str, default='./visualization_output', help='输出目录')
    parser.add_argument('--show', action='store_true', help='显示图表而不是保存')
    
    args = parser.parse_args()
    
    # 确定日志文件
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = find_latest_log_file(args.log_dir)
    
    # 创建可视化器
    visualizer = TrainingVisualizer(log_file)
    
    # 创建输出目录
    if not args.show:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成各种图表
        visualizer.plot_loss_curve(
            save_path=os.path.join(args.output_dir, f'loss_curve_{timestamp}.png')
        )
        visualizer.plot_learning_rate(
            save_path=os.path.join(args.output_dir, f'learning_rate_{timestamp}.png')
        )
        visualizer.plot_comprehensive_analysis(
            save_path=os.path.join(args.output_dir, f'comprehensive_analysis_{timestamp}.png')
        )
        
        # 生成统计报告
        visualizer.generate_statistics_report(
            save_path=os.path.join(args.output_dir, f'statistics_report_{timestamp}.txt')
        )
        
        print(f"\n🎉 所有可视化文件已保存到: {args.output_dir}")
    else:
        # 显示图表
        visualizer.plot_comprehensive_analysis()
        visualizer.generate_statistics_report()

if __name__ == "__main__":
    main()