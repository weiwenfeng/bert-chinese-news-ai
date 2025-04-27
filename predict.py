import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import numpy as np

def load_model_and_tokenizer():
    """加载模型和分词器"""
    # 加载模型
    model = BertForSequenceClassification.from_pretrained('models/bert_model')
    model.eval()  # 设置为评估模式
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('models/bert_tokenizer')
    
    # 加载标签字典
    with open('models/bert_label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    
    return model, tokenizer, label_dict

def predict_news(text, model, tokenizer, label_dict):
    """预测新闻类别"""
    # 对文本进行编码
    inputs = tokenizer(text, 
                      padding=True, 
                      truncation=True, 
                      max_length=512, 
                      return_tensors="pt")
    
    # 进行预测
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # 获取预测结果
    predicted_label = list(label_dict.keys())[list(label_dict.values()).index(predicted_class)]
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_label, confidence

def main():
    # 加载模型和分词器
    model, tokenizer, label_dict = load_model_and_tokenizer()
    
    # 更多不同类型的新闻文本
    example_texts = [
        # 体育新闻
        "中国女排在世界杯比赛中以3:0战胜美国队，取得五连胜。",
        "NBA总决赛：勇士队击败凯尔特人，夺得总冠军。",
        
        # 科技新闻
        "华为发布新一代5G芯片，性能提升30%。",
        "SpaceX成功发射新一代星链卫星，总数突破4000颗。",
        
        # 财经新闻
        "A股市场今日大涨，上证指数突破3500点。",
        "比特币价格突破6万美元，创历史新高。",
        
        # 娱乐新闻
        "第95届奥斯卡颁奖典礼：杨紫琼凭借《瞬息全宇宙》获最佳女主角。",
        "周杰伦新专辑《最伟大的作品》全球销量突破1000万张。",
        
        # 教育新闻
        "教育部：2023年高考报名人数达1291万，创历史新高。",
        "清华大学成立人工智能研究院，培养AI高端人才。",
        
        # 时政新闻
        "国务院召开常务会议，部署稳经济一揽子政策措施。",
        "外交部：中方坚决反对美方对台军售。",
        
        # 健康新闻
        "国家卫健委：全国新冠疫苗接种率超过90%。",
        "研究发现：每天运动30分钟可降低心脏病风险。",
        
        # 文化新闻
        "故宫博物院推出数字文物库，可在线欣赏3万件珍贵文物。",
        "中国传统节日端午节被列入联合国教科文组织人类非物质文化遗产。",
        
        # 国际新闻
        "联合国气候变化大会达成历史性协议，承诺减少碳排放。",
        "俄乌冲突持续，国际社会呼吁和平解决争端。",
        
        # 社会新闻
        "全国多地出现高温天气，气象部门发布高温预警。",
        "北京地铁新线开通，市民出行更加便捷。"
    ]
    
    # 对每个文本进行预测
    print("\n新闻分类预测结果：")
    print("-" * 80)
    for text in example_texts:
        label, confidence = predict_news(text, model, tokenizer, label_dict)
        print(f"文本：{text}")
        print(f"预测类别：{label}")
        print(f"置信度：{confidence:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main() 