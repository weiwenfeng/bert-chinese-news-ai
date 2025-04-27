import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import time
import pickle
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入工具函数
from utils.data_utils import prepare_data, clean_text
from utils.evaluation import evaluate_model, plot_confusion_matrix

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子
seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    logger.info("使用CPU")

def train_bert_model(data_path, stopwords_path=None, test_size=0.2, random_state=42, 
                     batch_size=8, epochs=4, learning_rate=2e-5):
    """
    使用预训练的中文BERT模型进行新闻分类
    """
    print("第四阶段：使用BERT模型进行新闻分类")
    
    # 加载数据
    print("加载和预处理数据...")
    df = prepare_data(data_path, stopwords_path=stopwords_path)
    
    # 查看数据集信息
    print(f"数据集大小: {df.shape}")
    print(f"类别分布:\n{df['category'].value_counts()}")
    
    # 为BERT准备数据
    # 使用clean_content而不是segmented_content，因为BERT有自己的tokenizer
    df['content_for_bert'] = df['clean_content']
    
    # 创建标签-ID映射
    labels = df['category'].unique()
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    
    # 将标签转换为ID
    df['label_id'] = df['category'].map(label_dict)
    
    # 保存标签字典
    with open('models/bert_label_dict.pkl', 'wb') as f:
        pickle.dump(label_dict, f)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['content_for_bert'], 
        df['label_id'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['label_id']
    )
    
    # 加载中文BERT tokenizer
    print("加载中文BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 保存tokenizer配置
    tokenizer.save_pretrained('models/bert_tokenizer')
    
    # 数据转换函数
    def convert_examples_to_features(texts, labels, tokenizer, max_length=128):
        input_ids = []
        attention_masks = []
        
        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,                        # 文本
                add_special_tokens=True,     # 添加 '[CLS]' 和 '[SEP]'
                max_length=max_length,       # 填充/截断长度
                padding='max_length',        # 填充到max_length
                return_attention_mask=True,  # 返回attention mask
                truncation=True,             # 截断超长文本
                return_tensors='pt',         # 返回PyTorch tensors
            )
            
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        # 转换为tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels.values)
        
        return input_ids, attention_masks, labels
    
    # 转换训练集
    print("转换数据为BERT输入格式...")
    train_inputs, train_masks, train_labels = convert_examples_to_features(
        X_train, y_train, tokenizer
    )
    
    # 转换测试集
    test_inputs, test_masks, test_labels = convert_examples_to_features(
        X_test, y_test, tokenizer
    )
    
    # 创建数据加载器
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    # 加载预训练的BERT模型
    print("加载并配置BERT模型...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels=len(label_dict),
        output_attentions=False,
        output_hidden_states=False,
    )
    
    model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    
    # 训练步数
    total_steps = len(train_dataloader) * epochs
    
    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练函数
    def train_epoch(model, dataloader, optimizer, scheduler, device):
        model.train()
        
        total_loss = 0
        total_accuracy = 0
        
        for batch in dataloader:
            # 从batch中获取数据
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            # 清零梯度
            model.zero_grad()
            
            # 前向传播
            outputs = model(
                batch_input_ids, 
                token_type_ids=None, 
                attention_mask=batch_input_mask, 
                labels=batch_labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # 反向传播
            loss.backward()
            
            # 计算准确率
            logits = logits.detach().cpu().numpy()
            label_ids = batch_labels.to('cpu').numpy()
            
            total_loss += loss.item()
            
            # 计算准确率
            preds = np.argmax(logits, axis=1).flatten()
            accuracy = np.sum(preds == label_ids) / len(label_ids)
            total_accuracy += accuracy
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        return avg_loss, avg_accuracy
    
    # 评估函数
    def evaluate(model, dataloader, device):
        model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            # 从batch中获取数据
            batch_input_ids = batch[0].to(device)
            batch_input_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)
            
            # 不计算梯度
            with torch.no_grad():
                # 前向传播
                outputs = model(
                    batch_input_ids, 
                    token_type_ids=None, 
                    attention_mask=batch_input_mask, 
                    labels=batch_labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # 转移到CPU
                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()
                
                # 保存预测结果
                preds = np.argmax(logits, axis=1).flatten()
                all_preds.extend(preds)
                all_labels.extend(label_ids)
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        
        # 计算准确率
        accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    # 训练模型
    print("开始训练BERT模型...")
    start_time = time.time()
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练一个epoch
        print("训练中...")
        train_loss, train_accuracy = train_epoch(
            model, train_dataloader, optimizer, scheduler, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"训练准确率: {train_accuracy:.4f}")
        
        # 评估模型
        print("评估中...")
        val_loss, val_accuracy, _, _ = evaluate(model, test_dataloader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
    
    training_time = time.time() - start_time
    print(f"\nBERT模型训练完成! 耗时: {training_time:.2f}秒")
    
    # 保存模型
    print("保存BERT模型...")
    model_save_path = 'models/bert_model'
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    
    # 在测试集上进行最终评估
    print("\n在测试集上进行最终评估...")
    start_time = time.time()
    test_loss, test_accuracy, test_preds, test_labels = evaluate(model, test_dataloader, device)
    evaluation_time = time.time() - start_time
    
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"评估时间: {evaluation_time:.2f}秒")
    
    # 将预测和标签映射回原始类别
    id_to_label = {v: k for k, v in label_dict.items()}
    pred_labels = [id_to_label[pred] for pred in test_preds]
    true_labels = [id_to_label[label] for label in test_labels]
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(true_labels, pred_labels, 
                         labels=labels, 
                         title='BERT模型的混淆矩阵')
    plt.savefig('stage4_bert_confusion_matrix.png')
    plt.close()
    
    print("\n混淆矩阵已保存为 stage4_bert_confusion_matrix.png")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, epochs+1), val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accuracies, 'b-', label='训练准确率')
    plt.plot(range(1, epochs+1), val_accuracies, 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('stage4_bert_learning_curve.png')
    plt.close()
    
    print("学习曲线已保存为 stage4_bert_learning_curve.png")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'label_dict': label_dict,
        'accuracy': test_accuracy,
        'training_time': training_time,
        'evaluation_time': evaluation_time
    }

def predict_with_bert(text, model, tokenizer, label_dict, device):
    """使用BERT模型预测文本类别"""
    # 清洗文本
    cleaned_text = clean_text(text)
    
    # 使用tokenizer处理文本
    encoded_dict = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )
    
    # 将输入移到设备上
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs.logits
    
    # 将logits转换为概率
    probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    # 获取预测的类别ID
    predicted_id = np.argmax(probs)
    
    # ID到标签的映射
    id_to_label = {v: k for k, v in label_dict.items()}
    predicted_category = id_to_label[predicted_id]
    
    # 获取预测概率
    category_probs = {id_to_label[i]: prob for i, prob in enumerate(probs)}
    
    return predicted_category, category_probs

if __name__ == "__main__":
    # 如果CUDA内存有限，可以设置较小的batch_size
    batch_size = 4
    
    # 运行下载数据脚本
    print("准备数据...")
    from download_data import download_thucnews_sample
    data_dir, stopwords_path = download_thucnews_sample()
    
    # 训练模型
    results = train_bert_model(data_dir, stopwords_path, batch_size=batch_size, epochs=2)
    
    # 测试预测功能
    print("\n测试预测功能:")
    test_texts = [
        "国家队比赛中，球员们发挥出色，以3-0的比分击败对手。",
        "股市今日大涨，上证指数上涨2.5%，创下近期新高。",
        "新款手机今日发布，搭载最新的处理器和先进的摄像头。"
    ]
    
    for text in test_texts:
        print(f"\n测试文本: {text}")
        
        # 使用BERT模型预测
        predicted_category, category_probs = predict_with_bert(
            text, results['model'], results['tokenizer'], results['label_dict'], device
        )
        print(f"BERT模型预测类别: {predicted_category}")
        top_3_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, prob in top_3_probs:
            print(f"  {cat}: {prob:.4f}")
    
    print("\n第四阶段完成！") 