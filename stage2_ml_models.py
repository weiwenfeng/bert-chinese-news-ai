import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils.data_utils import prepare_data
from utils.evaluation import evaluate_model, plot_confusion_matrix, plot_feature_importance

def train_traditional_ml_models(data_path, stopwords_path=None, test_size=0.2, random_state=42):
    """
    使用传统机器学习方法进行新闻分类
    """
    print("第二阶段：使用传统机器学习方法进行新闻分类")
    
    # 加载数据
    print("加载和预处理数据...")
    df = prepare_data(data_path, stopwords_path=stopwords_path)
    
    # 查看数据集信息
    print(f"数据集大小: {df.shape}")
    print(f"类别分布:\n{df['category'].value_counts()}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['segmented_content'], 
        df['category'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['category']
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # SVM模型
    print("\n训练SVM模型...")
    # 创建TF-IDF特征提取和SVM分类器的管道
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', SVC(kernel='linear', probability=True, random_state=random_state))
    ])
    
    # 使用交叉验证评估SVM模型
    print("执行交叉验证...")
    cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=5)
    print(f"交叉验证平均准确率: {np.mean(cv_scores):.4f} (std: {np.std(cv_scores):.4f})")
    
    # 训练最终SVM模型
    start_time = time.time()
    svm_pipeline.fit(X_train, y_train)
    svm_train_time = time.time() - start_time
    print(f"SVM训练时间: {svm_train_time:.2f}秒")
    
    # 在测试集上进行预测
    start_time = time.time()
    y_pred_svm = svm_pipeline.predict(X_test)
    svm_pred_time = time.time() - start_time
    print(f"SVM预测时间: {svm_pred_time:.2f}秒")
    
    # 评估SVM模型的性能
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM准确率: {svm_accuracy:.4f}")
    print("\nSVM分类报告:")
    print(classification_report(y_test, y_pred_svm))
    
    # 保存SVM模型
    with open('models/svm_pipeline.pkl', 'wb') as f:
        pickle.dump(svm_pipeline, f)
    
    # 随机森林模型
    print("\n训练随机森林模型...")
    # 创建TF-IDF特征提取和随机森林分类器的管道
    rf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])
    
    # 训练随机森林模型
    start_time = time.time()
    rf_pipeline.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    print(f"随机森林训练时间: {rf_train_time:.2f}秒")
    
    # 在测试集上进行预测
    start_time = time.time()
    y_pred_rf = rf_pipeline.predict(X_test)
    rf_pred_time = time.time() - start_time
    print(f"随机森林预测时间: {rf_pred_time:.2f}秒")
    
    # 评估随机森林模型的性能
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"随机森林准确率: {rf_accuracy:.4f}")
    print("\n随机森林分类报告:")
    print(classification_report(y_test, y_pred_rf))
    
    # 保存随机森林模型
    with open('models/rf_pipeline.pkl', 'wb') as f:
        pickle.dump(rf_pipeline, f)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(y_test, y_pred_svm, 
                         labels=np.unique(y_test), 
                         title='SVM模型的混淆矩阵')
    
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(y_test, y_pred_rf, 
                         labels=np.unique(y_test), 
                         title='随机森林模型的混淆矩阵')
    
    plt.tight_layout()
    plt.savefig('stage2_confusion_matrix.png')
    plt.close()
    
    print("\n混淆矩阵已保存为 stage2_confusion_matrix.png")
    
    # 特征重要性分析（随机森林）
    print("\n分析特征重要性...")
    # 获取特征名称
    feature_names = rf_pipeline.named_steps['tfidf'].get_feature_names_out()
    # 获取特征重要性
    feature_importance = rf_pipeline.named_steps['rf'].feature_importances_
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    plot_feature_importance(feature_importance, feature_names, top_n=20, 
                           title='随机森林模型的特征重要性')
    plt.savefig('stage2_feature_importance.png')
    plt.close()
    
    print("特征重要性已保存为 stage2_feature_importance.png")
    
    # 模型比较
    print("\n模型性能比较:")
    models_comparison = {
        'SVM': {
            'accuracy': svm_accuracy,
            'training_time': svm_train_time,
            'prediction_time': svm_pred_time
        },
        'RandomForest': {
            'accuracy': rf_accuracy,
            'training_time': rf_train_time,
            'prediction_time': rf_pred_time
        }
    }
    
    print(f"SVM准确率: {svm_accuracy:.4f}, 训练时间: {svm_train_time:.2f}秒, 预测时间: {svm_pred_time:.2f}秒")
    print(f"随机森林准确率: {rf_accuracy:.4f}, 训练时间: {rf_train_time:.2f}秒, 预测时间: {rf_pred_time:.2f}秒")
    
    return {
        'svm_pipeline': svm_pipeline,
        'rf_pipeline': rf_pipeline,
        'svm_accuracy': svm_accuracy,
        'rf_accuracy': rf_accuracy,
        'models_comparison': models_comparison
    }

def predict_with_ml_model(text, pipeline):
    """使用传统机器学习模型预测文本类别"""
    # 预测
    predicted_category = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    
    # 获取预测概率
    category_probs = {cat: prob for cat, prob in zip(pipeline.classes_, probabilities)}
    
    return predicted_category, category_probs

if __name__ == "__main__":
    # 运行下载数据脚本
    print("准备数据...")
    from download_data import download_thucnews_sample
    data_dir, stopwords_path = download_thucnews_sample()
    
    # 训练模型
    results = train_traditional_ml_models(data_dir, stopwords_path)
    
    # 测试预测功能
    print("\n测试预测功能:")
    test_texts = [
        "国家队比赛中，球员们发挥出色，以3-0的比分击败对手。",
        "股市今日大涨，上证指数上涨2.5%，创下近期新高。",
        "新款手机今日发布，搭载最新的处理器和先进的摄像头。"
    ]
    
    for text in test_texts:
        print(f"\n测试文本: {text}")
        
        # 使用SVM模型预测
        predicted_category, category_probs = predict_with_ml_model(
            text, results['svm_pipeline']
        )
        print(f"SVM模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
        
        # 使用随机森林模型预测
        predicted_category, category_probs = predict_with_ml_model(
            text, results['rf_pipeline']
        )
        print(f"随机森林模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
    
    print("\n第二阶段完成！") 