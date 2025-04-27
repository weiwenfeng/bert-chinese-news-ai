import os
import re
import jieba
import pandas as pd
from snownlp import SnowNLP

def load_stopwords(file_path):
    """加载停用词表"""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f])

def clean_text(text):
    """清洗文本"""
    # 去除HTML标签
    text = re.sub('<.*?>', '', text)
    # 去除URL
    text = re.sub(r'http\S+', '', text)
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 去除英文字符
    text = re.sub(r'[a-zA-Z]', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def segment_text(text, stopwords=None):
    """分词"""
    words = jieba.cut(text)
    if stopwords:
        words = [word for word in words if word not in stopwords and word.strip()]
    return ' '.join(words)

def prepare_data(data_path, categories=None, sample_size=None, stopwords_path=None):
    """
    准备数据集
    :param data_path: 数据集路径
    :param categories: 要处理的类别列表
    :param sample_size: 每个类别的样本数量
    :param stopwords_path: 停用词表路径
    :return: 处理后的数据集
    """
    stopwords = load_stopwords(stopwords_path) if stopwords_path else set()
    data = []
    
    if os.path.isfile(data_path) and data_path.endswith('.csv'):
        # 如果是CSV文件
        df = pd.read_csv(data_path)
        if 'content' in df.columns and 'category' in df.columns:
            df['clean_content'] = df['content'].apply(clean_text)
            df['segmented_content'] = df['clean_content'].apply(lambda x: segment_text(x, stopwords))
            return df
    else:
        # 如果是文件夹结构
        for category in os.listdir(data_path):
            if categories and category not in categories:
                continue
                
            category_path = os.path.join(data_path, category)
            if not os.path.isdir(category_path):
                continue
                
            files = os.listdir(category_path)
            if sample_size:
                files = files[:min(sample_size, len(files))]
                
            for file in files:
                file_path = os.path.join(category_path, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    clean_content = clean_text(content)
                    segmented_content = segment_text(clean_content, stopwords)
                    data.append({
                        'category': category,
                        'content': content,
                        'clean_content': clean_content,
                        'segmented_content': segmented_content
                    })
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    
    return pd.DataFrame(data)

def get_sentiment(text):
    """使用SnowNLP获取情感分析"""
    try:
        s = SnowNLP(text)
        return s.sentiments
    except:
        return 0.5  # 默认中性

def get_keywords(text, num=5):
    """提取关键词"""
    try:
        s = SnowNLP(text)
        return s.keywords(num)
    except:
        return []

def get_summary(text, num=3):
    """生成摘要"""
    try:
        s = SnowNLP(text)
        return s.summary(num)
    except:
        return [] 