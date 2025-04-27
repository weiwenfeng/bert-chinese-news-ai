import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils.data_utils import prepare_data
from utils.evaluation import evaluate_model, plot_learning_curve

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

def train_deep_learning_models(data_path, stopwords_path=None, test_size=0.2, random_state=42):
    """
    使用深度学习模型进行新闻分类
    """
    print("第三阶段：使用深度学习模型进行新闻分类")
    
    # 加载数据
    print("加载和预处理数据...")
    df = prepare_data(data_path, stopwords_path=stopwords_path)
    
    # 查看数据集信息
    print(f"数据集大小: {df.shape}")
    print(f"类别分布:\n{df['category'].value_counts()}")
    
    # 对类别标签进行编码
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['category'])
    num_classes = len(label_encoder.classes_)
    print(f"类别数量: {num_classes}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['segmented_content'], 
        df['category_encoded'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['category_encoded']
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 文本转换为序列
    max_words = 10000  # 词汇表大小
    max_len = 200      # 序列最大长度
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # 保存tokenizer
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # 保存标签编码器
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # 保存配置
    config = {
        'max_words': max_words,
        'max_len': max_len
    }
    with open('models/dl_config.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    # 创建并训练CNN模型
    print("\n训练CNN模型...")
    cnn_model = build_cnn_model(max_words, max_len, num_classes)
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
    
    start_time = time.time()
    cnn_history = cnn_model.fit(
        X_train_pad, tf.keras.utils.to_categorical(y_train, num_classes),
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    cnn_train_time = time.time() - start_time
    print(f"CNN训练时间: {cnn_train_time:.2f}秒")
    
    # 在测试集上评估CNN模型
    start_time = time.time()
    cnn_scores = cnn_model.evaluate(X_test_pad, tf.keras.utils.to_categorical(y_test, num_classes))
    cnn_pred_time = time.time() - start_time
    print(f"CNN评估时间: {cnn_pred_time:.2f}秒")
    print(f"CNN测试损失: {cnn_scores[0]:.4f}")
    print(f"CNN测试准确率: {cnn_scores[1]:.4f}")
    
    # 保存CNN模型
    cnn_model.save('models/cnn_model.h5')
    
    # 绘制CNN学习曲线
    plt.figure(figsize=(12, 5))
    plot_learning_curve(cnn_history)
    plt.savefig('stage3_cnn_learning_curve.png')
    plt.close()
    print("CNN学习曲线已保存为 stage3_cnn_learning_curve.png")
    
    # 创建并训练LSTM模型
    print("\n训练LSTM模型...")
    lstm_model = build_lstm_model(max_words, max_len, num_classes)
    
    start_time = time.time()
    lstm_history = lstm_model.fit(
        X_train_pad, tf.keras.utils.to_categorical(y_train, num_classes),
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )
    lstm_train_time = time.time() - start_time
    print(f"LSTM训练时间: {lstm_train_time:.2f}秒")
    
    # 在测试集上评估LSTM模型
    start_time = time.time()
    lstm_scores = lstm_model.evaluate(X_test_pad, tf.keras.utils.to_categorical(y_test, num_classes))
    lstm_pred_time = time.time() - start_time
    print(f"LSTM评估时间: {lstm_pred_time:.2f}秒")
    print(f"LSTM测试损失: {lstm_scores[0]:.4f}")
    print(f"LSTM测试准确率: {lstm_scores[1]:.4f}")
    
    # 保存LSTM模型
    lstm_model.save('models/lstm_model.h5')
    
    # 绘制LSTM学习曲线
    plt.figure(figsize=(12, 5))
    plot_learning_curve(lstm_history)
    plt.savefig('stage3_lstm_learning_curve.png')
    plt.close()
    print("LSTM学习曲线已保存为 stage3_lstm_learning_curve.png")
    
    # 模型比较
    print("\n深度学习模型性能比较:")
    models_comparison = {
        'CNN': {
            'accuracy': cnn_scores[1],
            'training_time': cnn_train_time,
            'prediction_time': cnn_pred_time
        },
        'LSTM': {
            'accuracy': lstm_scores[1],
            'training_time': lstm_train_time,
            'prediction_time': lstm_pred_time
        }
    }
    
    print(f"CNN准确率: {cnn_scores[1]:.4f}, 训练时间: {cnn_train_time:.2f}秒, 预测时间: {cnn_pred_time:.2f}秒")
    print(f"LSTM准确率: {lstm_scores[1]:.4f}, 训练时间: {lstm_train_time:.2f}秒, 预测时间: {lstm_pred_time:.2f}秒")
    
    return {
        'cnn_model': cnn_model,
        'lstm_model': lstm_model,
        'tokenizer': tokenizer,
        'label_encoder': label_encoder,
        'config': config,
        'cnn_accuracy': cnn_scores[1],
        'lstm_accuracy': lstm_scores[1],
        'models_comparison': models_comparison
    }

def build_cnn_model(max_words, max_len, num_classes, embedding_dim=100):
    """
    构建CNN模型
    """
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def build_lstm_model(max_words, max_len, num_classes, embedding_dim=100):
    """
    构建LSTM模型
    """
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def predict_with_dl_model(text, model, tokenizer, label_encoder, max_len):
    """使用深度学习模型预测文本类别"""
    # 文本转换为序列
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    # 预测
    prediction = model.predict(padded_sequence)[0]
    predicted_index = np.argmax(prediction)
    predicted_category = label_encoder.inverse_transform([predicted_index])[0]
    
    # 获取预测概率
    category_probs = {cat: prediction[i] for i, cat in enumerate(label_encoder.classes_)}
    
    return predicted_category, category_probs

if __name__ == "__main__":
    # 运行下载数据脚本
    print("准备数据...")
    from download_data import download_thucnews_sample
    data_dir, stopwords_path = download_thucnews_sample()
    
    # 训练模型
    results = train_deep_learning_models(data_dir, stopwords_path)
    
    # 测试预测功能
    print("\n测试预测功能:")
    test_texts = [
        "国家队比赛中，球员们发挥出色，以3-0的比分击败对手。",
        "股市今日大涨，上证指数上涨2.5%，创下近期新高。",
        "新款手机今日发布，搭载最新的处理器和先进的摄像头。"
    ]
    
    # 从结果中获取必要的组件
    cnn_model = results['cnn_model']
    lstm_model = results['lstm_model']
    tokenizer = results['tokenizer']
    label_encoder = results['label_encoder']
    max_len = results['config']['max_len']
    
    for text in test_texts:
        print(f"\n测试文本: {text}")
        
        # 使用CNN模型预测
        predicted_category, category_probs = predict_with_dl_model(
            text, cnn_model, tokenizer, label_encoder, max_len
        )
        print(f"CNN模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
        
        # 使用LSTM模型预测
        predicted_category, category_probs = predict_with_dl_model(
            text, lstm_model, tokenizer, label_encoder, max_len
        )
        print(f"LSTM模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
    
    print("\n第三阶段完成！") 