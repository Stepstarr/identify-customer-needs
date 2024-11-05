import pandas as pd
import re

import re
import csv
from pathlib import Path
from collections import Counter
from dataset import MyToken, MySentence, MyImage, MyPair, MyDataset, MyCorpus
# import constants
from PIL import Image
import torch

# constants for preprocessing
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
IMGID_PREFIX = 'IMGID:'
URL_PREFIX = 'http://t.co/'
UNKNOWN_TOKEN = '[UNK]'


def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text

# 这里调取测试集训练集不随机
def load_itr_corpus(path: str, split: int = 3576, normalize: bool = False):
    path = Path(path)
    path_to_images = path / 'Twitter_images'
    assert path.exists()
    assert path_to_images.exists()
    
    with open(path/'data2.csv', encoding='utf-8',newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
      
        pairs = [MyPair(
            sentence=MySentence(text=normalize_text(row['cleaned_tweet']) if normalize else row['tweet']),
            image=MyImage(row['Matched_Image']),
            image_explan=MySentence(text=normalize_text(row['image_explanation']) if normalize else row['image_explanation']),
            label=int(row['image_adds'])
            # label=int(row['image_notadds_text_notrepr'])
        ) for row in csv_reader]

    train = MyDataset(pairs[:split], path_to_images)
    test = MyDataset(pairs[split:], path_to_images)
    return MyCorpus(train=train, test=test)


def load_ner_dataset(path_to_txt: Path, path_to_images: Path, load_image: bool = True) -> MyDataset:
    tokens = []
    image_id = None
    pairs = []

    with open(str(path_to_txt), encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.rstrip()  # strip '\n'

            if line.startswith(IMGID_PREFIX):
                image_id = line[len(IMGID_PREFIX):]
            elif line != '':
                text, label = line.split('\t')
                if text == '' or text.isspace() \
                        or text in SPECIAL_TOKENS \
                        or text.startswith(URL_PREFIX):
                    text = UNKNOWN_TOKEN
                tokens.append(MyToken(text, constants.LABEL_TO_ID[label]))
            else:
                pairs.append(MyPair(MySentence(tokens), MyImage(f'{image_id}.jpg')))
                tokens = []
    pairs.append(MyPair(MySentence(tokens), MyImage(f'{image_id}.jpg')))

    return MyDataset(pairs, path_to_images, load_image)


def load_ner_corpus(path: str, load_image: bool = True) -> MyCorpus:
    path = Path(path)
    path_to_train_file = path / 'train.txt'
    path_to_dev_file = path / 'dev.txt'
    path_to_test_file = path / 'test.txt'
    path_to_images = path / 'images'

    assert path_to_train_file.exists()
    assert path_to_dev_file.exists()
    assert path_to_test_file.exists()
    assert path_to_images.exists()

    train = load_ner_dataset(path_to_train_file, path_to_images, load_image)
    dev = load_ner_dataset(path_to_dev_file, path_to_images, load_image)
    test = load_ner_dataset(path_to_test_file, path_to_images, load_image)

    return MyCorpus(train, dev, test)


def type_count(dataset: MyDataset) -> str:
    tags = [token.label for pair in dataset for token in pair.sentence]
    counter = Counter(tags)

    num_total = len(dataset)
    num_per = counter['B-PER']
    num_loc = counter['B-LOC']
    num_org = counter['B-ORG']
    num_misc = counter['B-MISC']

    return f'{num_total}\t{num_per}\t{num_loc}\t{num_org}\t{num_misc}'


def token_count(dataset: MyDataset) -> str:
    lengths = [len(pair.sentence) for pair in dataset]

    num_sentences = len(lengths)
    num_tokens = sum(lengths)

    return f'{num_sentences}\t{num_tokens}'


def print_image_info(image):
    print(f"  图片对象类型: {type(image)}")
    if hasattr(image, 'file_name'):
        print(f"    文件名: {image.file_name}")
    if hasattr(image, 'data'):
        print(f"    数据类型: {type(image.data)}")
        if isinstance(image.data, Image.Image):
            print(f"      图片模式: {image.data.mode}")
            print(f"      图片大小: {image.data.size}")
        elif isinstance(image.data, torch.Tensor):
            print(f"      张量形状: {image.data.shape}")
            print(f"      张量类型: {image.data.dtype}")
    print(f"    其他属性: {[attr for attr in dir(image) if not attr.startswith('__') and attr not in ['file_name', 'data']]}")

if __name__ == "__main__":
    # 导入数据
    trc = load_itr_corpus('picture_text_relation/TRC_data')
    print("TRC 数据集信息:")
    print(f"trc 的类型: {type(trc)}")
    print(f"trc.train 的类型: {type(trc.train)}")
    print(f"trc.train 的长度: {len(trc.train)}")

    print("\n尝试打印前5个样本:")
    for i, item in enumerate(trc.train):
        if i >= 5:
            break
        print(f"\n样本 {i+1}:")
        print(f"类型: {type(item)}")
        if hasattr(item, 'sentence'):
            print(f"  句子: {item.sentence}")
            if hasattr(item.sentence, 'text'):
                print(f"    文本: {item.sentence.text}")
        if hasattr(item, 'image'):
            print("  图片信息:")
            print_image_info(item.image)
        if hasattr(item, 'image_explan'):
            print(f"  图片解释: {item.image_explan}")
            if hasattr(item.image_explan, 'text'):
                print(f"    文本: {item.image_explan.text}")
        if hasattr(item, 'label'):
            print(f"  标签: {item.label}")

    # 如果 trc.train 是 MyDataset 类型，尝试直接访问其 pairs 属性
    if hasattr(trc.train, 'pairs'):
        print("\ntrc.train.pairs 的信息:")
        print(f"类型: {type(trc.train.pairs)}")
        print(f"长度: {len(trc.train.pairs)}")
        if len(trc.train.pairs) > 0:
            print("\n第一个 pair 的信息:")
            first_pair = trc.train.pairs[0]
            print(f"类型: {type(first_pair)}")
            print(f"内容: {first_pair}")
            if hasattr(first_pair, 'sentence'):
                print(f"  句子: {first_pair.sentence}")
                if hasattr(first_pair.sentence, 'text'):
                    print(f"    文本: {first_pair.sentence.text}")
            if hasattr(first_pair, 'image'):
                print(f"  图片: {first_pair.image}")
                if hasattr(first_pair.image, 'path'):
                    print(f"    路径: {first_pair.image.path}")
                # 打印 MyImage 对象的所有属性
                print(f"    MyImage 属性: {vars(first_pair.image)}")
            if hasattr(first_pair, 'image_explan'):
                print(f"  图片解释: {first_pair.image_explan}")
                if hasattr(first_pair.image_explan, 'text'):
                    print(f"    文本: {first_pair.image_explan.text}")
            if hasattr(first_pair, 'label'):
                print(f"  标签: {first_pair.label}")

   