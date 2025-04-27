import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from snownlp import SnowNLP
import jieba
import pickle

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils.data_utils import prepare_data, clean_text, segment_text, load_stopwords
from utils.evaluation import evaluate_model, plot_confusion_matrix

def train_snownlp_naive_bayes(data_path, stopwords_path=None, test_size=0.2, random_state=42):
    """
    使用SnowNLP特性和朴素贝叶斯分类器进行新闻分类
    """
    print("第一阶段：使用SnowNLP和朴素贝叶斯进行新闻分类")
    
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
    
    # 特征提取：词袋模型
    print("使用词袋模型提取特征...")
    count_vectorizer = CountVectorizer()
    X_train_counts = count_vectorizer.fit_transform(X_train)
    X_test_counts = count_vectorizer.transform(X_test)
    
    # 训练朴素贝叶斯分类器
    print("训练朴素贝叶斯分类器...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_counts, y_train)
    
    # 在测试集上进行预测
    y_pred_counts = nb_classifier.predict(X_test_counts)
    
    # 评估词袋模型 + 朴素贝叶斯的性能
    bow_accuracy = accuracy_score(y_test, y_pred_counts)
    print(f"词袋模型 + 朴素贝叶斯的准确率: {bow_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_counts))
    
    # 保存词袋模型和分类器
    os.makedirs('models', exist_ok=True)
    with open('models/count_vectorizer.pkl', 'wb') as f:
        pickle.dump(count_vectorizer, f)
    with open('models/nb_classifier_bow.pkl', 'wb') as f:
        pickle.dump(nb_classifier, f)
    
    # 特征提取：TF-IDF
    print("\n使用TF-IDF提取特征...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # 训练朴素贝叶斯分类器 (TF-IDF特征)
    print("训练朴素贝叶斯分类器 (TF-IDF特征)...")
    nb_classifier_tfidf = MultinomialNB()
    nb_classifier_tfidf.fit(X_train_tfidf, y_train)
    
    # 在测试集上进行预测
    y_pred_tfidf = nb_classifier_tfidf.predict(X_test_tfidf)
    
    # 评估TF-IDF模型 + 朴素贝叶斯的性能
    tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
    print(f"TF-IDF + 朴素贝叶斯的准确率: {tfidf_accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_tfidf))
    
    # 保存TF-IDF模型和分类器
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open('models/nb_classifier_tfidf.pkl', 'wb') as f:
        pickle.dump(nb_classifier_tfidf, f)
    
    # 使用SnowNLP分析情感和关键词
    print("\n使用SnowNLP进行文本分析...")
    sentiment_scores = []
    keywords_lists = []
    
    # 抽取一部分样本进行演示
    sample_texts = X_test.iloc[:5].values
    sample_categories = y_test.iloc[:5].values
    
    for i, text in enumerate(sample_texts):
        try:
            s = SnowNLP(text)
            sentiment = s.sentiments
            keywords = s.keywords(5)  # 提取5个关键词
            
            sentiment_scores.append(sentiment)
            keywords_lists.append(keywords)
            
            print(f"\n样本 {i+1} (类别: {sample_categories[i]}):")
            print(f"文本: {text[:100]}...")
            print(f"情感分数: {sentiment:.4f} (越接近1越正面)")
            print(f"关键词: {', '.join(keywords)}")
            print(f"TF-IDF预测类别: {y_pred_tfidf[i]}")
        except Exception as e:
            print(f"处理样本 {i+1} 时出错: {e}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    cm_bow = plot_confusion_matrix(y_test, y_pred_counts, 
                                  labels=nb_classifier.classes_, 
                                  title='词袋模型 + 朴素贝叶斯的混淆矩阵')
    
    plt.subplot(1, 2, 2)
    cm_tfidf = plot_confusion_matrix(y_test, y_pred_tfidf, 
                                    labels=nb_classifier_tfidf.classes_, 
                                    title='TF-IDF + 朴素贝叶斯的混淆矩阵')
    
    plt.tight_layout()
    plt.savefig('stage1_confusion_matrix.png')
    plt.close()
    
    print("\n混淆矩阵已保存为 stage1_confusion_matrix.png")
    
    return {
        'bow_accuracy': bow_accuracy,
        'tfidf_accuracy': tfidf_accuracy,
        'bow_model': nb_classifier,
        'tfidf_model': nb_classifier_tfidf,
        'count_vectorizer': count_vectorizer,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def predict_category(text, vectorizer, model):
    """使用训练好的模型预测文本类别"""
    # 文本预处理
    cleaned_text = clean_text(text)
    segmented_text = segment_text(cleaned_text)
    
    # 特征提取
    text_features = vectorizer.transform([segmented_text])
    
    # 预测
    predicted_category = model.predict(text_features)[0]
    probabilities = model.predict_proba(text_features)[0]
    
    # 获取预测概率
    category_probs = {cat: prob for cat, prob in zip(model.classes_, probabilities)}
    
    return predicted_category, category_probs

if __name__ == "__main__":
    # 运行下载数据脚本
    print("准备数据...")
    from download_data import download_thucnews_sample
    data_dir, stopwords_path = download_thucnews_sample()
    
    # 训练模型
    results = train_snownlp_naive_bayes(data_dir, stopwords_path)
    
    # 测试预测功能
    print("\n测试预测功能:")
    test_texts = [
        "国家队比赛中，球员们发挥出色，以3-0的比分击败对手。",
        "股市今日大涨，上证指数上涨2.5%，创下近期新高。",
        "新款手机今日发布，搭载最新的处理器和先进的摄像头。"
    ]
    
    for text in test_texts:
        print(f"\n测试文本: {text}")
        
        # 使用词袋模型预测
        predicted_category, category_probs = predict_category(
            text, results['count_vectorizer'], results['bow_model']
        )
        print(f"词袋模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
        
        # 使用TF-IDF模型预测
        predicted_category, category_probs = predict_category(
            text, results['tfidf_vectorizer'], results['tfidf_model']
        )
        print(f"TF-IDF模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
    
    print("\n第一阶段完成！") 