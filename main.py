import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="中文新闻分类系统")
    parser.add_argument('--stage', type=int, default=0, 
                       help='要运行的阶段 (0=全部, 1=SnowNLP基础分类, 2=传统机器学习, 3=深度学习, 4=BERT)')
    parser.add_argument('--download', action='store_true', help='是否重新下载/准备数据')
    args = parser.parse_args()
    
    # 准备数据
    if args.download or not os.path.exists(os.path.join('data', 'thucnews_sample')):
        print("准备数据...")
        from download_data import download_thucnews_sample
        data_dir, stopwords_path = download_thucnews_sample()
    else:
        data_dir = os.path.join('data', 'thucnews_sample')
        stopwords_path = os.path.join('data', 'stopwords.txt')
    
    # 创建models目录
    os.makedirs('models', exist_ok=True)
    
    # 结果存储
    all_results = {}
    
    # 运行阶段1：SnowNLP基础分类
    if args.stage == 0 or args.stage == 1:
        print("\n" + "="*50)
        print("阶段1：SnowNLP基础分类")
        print("="*50)
        
        start_time = time.time()
        from stage1_snownlp_basic import train_snownlp_naive_bayes
        stage1_results = train_snownlp_naive_bayes(data_dir, stopwords_path)
        stage1_time = time.time() - start_time
        
        all_results['stage1'] = {
            'bow_accuracy': stage1_results['bow_accuracy'],
            'tfidf_accuracy': stage1_results['tfidf_accuracy'],
            'time': stage1_time
        }
        
        print(f"\n阶段1完成! 耗时: {stage1_time:.2f}秒")
    
    # 运行阶段2：传统机器学习方法
    if args.stage == 0 or args.stage == 2:
        print("\n" + "="*50)
        print("阶段2：传统机器学习方法")
        print("="*50)
        
        start_time = time.time()
        from stage2_ml_models import train_traditional_ml_models
        stage2_results = train_traditional_ml_models(data_dir, stopwords_path)
        stage2_time = time.time() - start_time
        
        all_results['stage2'] = {
            'svm_accuracy': stage2_results['svm_accuracy'],
            'rf_accuracy': stage2_results['rf_accuracy'],
            'time': stage2_time
        }
        
        print(f"\n阶段2完成! 耗时: {stage2_time:.2f}秒")
    
    # 运行阶段3：深度学习模型
    if args.stage == 0 or args.stage == 3:
        print("\n" + "="*50)
        print("阶段3：深度学习模型")
        print("="*50)
        
        start_time = time.time()
        from stage3_deep_learning import train_deep_learning_models
        stage3_results = train_deep_learning_models(data_dir, stopwords_path)
        stage3_time = time.time() - start_time
        
        all_results['stage3'] = {
            'cnn_accuracy': stage3_results['cnn_accuracy'],
            'lstm_accuracy': stage3_results['lstm_accuracy'],
            'time': stage3_time
        }
        
        print(f"\n阶段3完成! 耗时: {stage3_time:.2f}秒")
    
    # 运行阶段4：BERT模型
    if args.stage == 0 or args.stage == 4:
        print("\n" + "="*50)
        print("阶段4：BERT模型")
        print("="*50)
        
        start_time = time.time()
        from stage4_bert import train_bert_model
        # 如果显存有限，可以调整batch_size和epochs
        stage4_results = train_bert_model(data_dir, stopwords_path, batch_size=4, epochs=2)
        stage4_time = time.time() - start_time
        
        all_results['stage4'] = {
            'bert_accuracy': stage4_results['accuracy'],
            'time': stage4_time
        }
        
        print(f"\n阶段4完成! 耗时: {stage4_time:.2f}秒")
    
    # 输出所有结果对比
    if args.stage == 0:
        print("\n" + "="*50)
        print("所有模型性能对比")
        print("="*50)
        
        accuracies = []
        times = []
        model_names = []
        
        if 'stage1' in all_results:
            model_names.extend(['词袋+朴素贝叶斯', 'TF-IDF+朴素贝叶斯'])
            accuracies.extend([all_results['stage1']['bow_accuracy'], 
                              all_results['stage1']['tfidf_accuracy']])
            times.extend([all_results['stage1']['time']/2, all_results['stage1']['time']/2])
        
        if 'stage2' in all_results:
            model_names.extend(['SVM', '随机森林'])
            accuracies.extend([all_results['stage2']['svm_accuracy'], 
                              all_results['stage2']['rf_accuracy']])
            times.extend([all_results['stage2']['time']/2, all_results['stage2']['time']/2])
        
        if 'stage3' in all_results:
            model_names.extend(['CNN', 'LSTM'])
            accuracies.extend([all_results['stage3']['cnn_accuracy'], 
                              all_results['stage3']['lstm_accuracy']])
            times.extend([all_results['stage3']['time']/2, all_results['stage3']['time']/2])
        
        if 'stage4' in all_results:
            model_names.append('BERT')
            accuracies.append(all_results['stage4']['bert_accuracy'])
            times.append(all_results['stage4']['time'])
        
        # 打印结果表格
        print("\n精度对比:")
        print("-"*60)
        print(f"{'模型':<20} {'准确率':<15} {'训练时间(秒)':<15}")
        print("-"*60)
        for name, acc, t in zip(model_names, accuracies, times):
            print(f"{name:<20} {acc:<15.4f} {t:<15.2f}")
        print("-"*60)
        
        # 绘制准确率对比图
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(model_names, accuracies, color='skyblue')
        plt.title('模型准确率对比')
        plt.xlabel('模型')
        plt.ylabel('准确率')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.subplot(1, 2, 2)
        bars = plt.bar(model_names, times, color='lightgreen')
        plt.title('模型训练时间对比')
        plt.xlabel('模型')
        plt.ylabel('时间(秒)')
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig('models_comparison.png')
        plt.close()
        
        print("\n模型对比图已保存为 models_comparison.png")
    
    print("\n项目执行完毕!")

if __name__ == "__main__":
    main()