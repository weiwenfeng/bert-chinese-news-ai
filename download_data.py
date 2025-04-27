import os
import urllib.request
import pandas as pd
import tarfile
import zipfile
import random
import shutil

def download_thucnews_sample():
    """
    下载THUCNews数据集的子集
    由于完整的THUCNews数据集比较大，我们创建一个小的样本数据集用于演示
    """
    print("正在创建示例数据...")
    
    # 创建数据目录
    data_dir = os.path.join('data', 'thucnews_sample')
    os.makedirs(data_dir, exist_ok=True)
    
    # 类别列表
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    
    # 为每个类别创建一些示例文本
    sample_texts = {
        '体育': [
            "国足昨日在世界杯预选赛中以2-0战胜了对手，取得了重要的3分。",
            "NBA总决赛第四场，勇士队以107-97击败猛龙队，将总比分扳成2-2。",
            "刘翔在110米栏比赛中跑出了13秒11的好成绩，获得本站钻石联赛冠军。"
        ],
        '财经': [
            "央行今日宣布降息0.25个百分点，旨在刺激经济增长。",
            "上证指数今日上涨1.2%，创下近两个月来的新高。",
            "多家科技公司公布财报，营收普遍超出市场预期。"
        ],
        '房产': [
            "北京房价近期出现小幅回调，环比下降1.5%。",
            "住建部发布新政策，加强对房地产市场的监管。",
            "多地松绑限购政策，释放楼市活力。"
        ],
        '家居': [
            "简约风格在今年的家居设计中非常流行，强调空间的实用性。",
            "如何选择适合小户型的家具？专家给出了这些建议。",
            "智能家居系统让生活更便捷，市场规模不断扩大。"
        ],
        '教育': [
            "教育部公布新的课程改革方案，将增加编程教育的比重。",
            "高考改革持续推进，多地启动新的招生模式。",
            "在线教育平台用户数量激增，行业竞争加剧。"
        ],
        '科技': [
            "华为发布最新旗舰手机，搭载自研芯片和创新摄像技术。",
            "SpaceX成功发射猎鹰9号火箭，将60颗星链卫星送入轨道。",
            "人工智能技术在医疗领域的应用不断深入，助力疾病诊断。"
        ],
        '时尚': [
            "巴黎时装周落幕，可持续时尚成为本季的主题。",
            "时尚博主分享夏季穿搭技巧，轻薄面料成为首选。",
            "奢侈品牌纷纷进军中国市场，线上销售渠道表现亮眼。"
        ],
        '时政': [
            "国家主席出访欧洲三国，深化经贸合作。",
            "两会期间，代表委员就经济发展提出多项建议。",
            "政府工作报告强调，今年经济增长目标定为6%左右。"
        ],
        '游戏': [
            "《英雄联盟》全球总决赛开幕，16支战队角逐冠军。",
            "索尼PS5发售首周销量突破100万台，创下历史记录。",
            "国产游戏海外收入持续增长，文化输出效应明显。"
        ],
        '娱乐': [
            "第93届奥斯卡颁奖典礼举行，《无依之地》获最佳影片。",
            "流行歌手新专辑发行首周，数字销量突破百万。",
            "热播剧收官，豆瓣评分9.2，成为年度最佳剧集之一。"
        ]
    }
    
    # 为每个类别创建目录和样本文件
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        texts = sample_texts[category]
        # 为了增加样本量，我们复制几份
        for i in range(10):
            for j, text in enumerate(texts):
                # 稍微变化一下内容
                modified_text = text + f" 这是第{i+1}版的报道。"
                file_path = os.path.join(category_dir, f"{i}_{j}.txt")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_text)
    
    print(f"示例数据已生成在 {data_dir} 目录下")
    print(f"共创建了 {len(categories)} 个类别，每个类别包含 {len(texts) * 10} 个样本")
    
    # 创建停用词文件
    stopwords = [
        "的", "了", "和", "是", "在", "我", "有", "这", "个", "那", "你", "们", "就", "也", "都",
        "要", "为", "以", "到", "等", "着", "又", "或", "并", "很", "会", "可以", "没有", "不是"
    ]
    
    stopwords_path = os.path.join('data', 'stopwords.txt')
    with open(stopwords_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stopwords))
    
    print(f"停用词表已生成在 {stopwords_path}")
    
    return data_dir, stopwords_path

if __name__ == "__main__":
    data_dir, stopwords_path = download_thucnews_sample()
    print("数据准备完成！") 