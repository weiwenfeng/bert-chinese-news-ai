import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties

# 设置中文字体，默认使用系统自带的中文字体
try:
    font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
except:
    font = None

def plot_confusion_matrix(y_true, y_pred, labels=None, title='混淆矩阵'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('预测类别', fontproperties=font)
    plt.ylabel('真实类别', fontproperties=font)
    plt.title(title, fontproperties=font)
    plt.tight_layout()
    return plt

def evaluate_model(y_true, y_pred, labels=None):
    """评估模型性能"""
    # 计算总体准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算每个类的精确率、召回率、F1值
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels)
    
    # 生成分类报告
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels)
    
    # 返回评估结果
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'report': report
    }

def plot_learning_curve(history, metrics=['loss', 'accuracy']):
    """绘制学习曲线"""
    plt.figure(figsize=(12, 5))
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        plt.plot(history.history[metric], label=f'训练{metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'验证{metric}')
        plt.title(f'{metric}曲线', fontproperties=font)
        plt.xlabel('Epoch', fontproperties=font)
        plt.ylabel(metric, fontproperties=font)
        plt.legend(prop=font)
    
    plt.tight_layout()
    return plt

def plot_model_comparison(models_results, title='模型性能对比'):
    """绘制不同模型之间的性能对比"""
    models = list(models_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # 准备数据
    data = {metric: [models_results[model][metric] for model in models] for metric in metrics}
    df = pd.DataFrame(data, index=models)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(title, fontproperties=font)
    plt.xlabel('模型', fontproperties=font)
    plt.ylabel('分数', fontproperties=font)
    plt.xticks(rotation=45)
    plt.legend(prop=font)
    plt.tight_layout()
    
    return plt

def plot_feature_importance(feature_importance, feature_names, top_n=20, title='特征重要性'):
    """绘制特征重要性"""
    # 创建特征重要性的数据框
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    
    # 按重要性排序并选择前N个特征
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(title, fontproperties=font)
    plt.xlabel('重要性', fontproperties=font)
    plt.ylabel('特征', fontproperties=font)
    plt.tight_layout()
    
    return plt 