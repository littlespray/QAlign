import json
import math
import os
import copy

# 假设txt文件的路径为'data.txt'
# txt_file_path = '/userhome/crave-db/shuffle_final_prompt_train.txt'
# json_file_path = '/userhome/crave-db/shuffle_final_prompt_train.json'

# txt_file_path = '/userhome/hunyuan_video2.txt'
# json_file_path = '/userhome/hunyuan-test.json'

txt_file_path = './IE-R1-4K/train.txt'
json_file_path = './IE-R11-train-dual.json'

# txt_file_path = './IE-R1-4K/val.txt'
# json_file_path = './IE-R1-val-traindst.json'

# 初始化一个空列表来存储字典
data_list = []

# 读取txt文件
with open(txt_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行尾的换行符并分割字符串
        parts = line.strip().split('|')
        if len(parts) >= 4:
            name = parts[0]  # img_basename
            instruct = parts[1]  # instruct
            # parts[2] = src_caption, parts[3] = dst_caption
            label = parts[-1]
            # label=float(label) / 20 # T2V-QA
            # label=float(label)/2 # 如果是1-10分就需要除以2
            label = float(label)
            # 创建一个字典并添加到列表中
            data_dict = {
                "name": name,
                "gt_score": label,
                "instruction": instruct,
                # 为 train_mem.py 的 dual_image_mode 准备：同时提供 src/edited 字段与 image 列表
                "src_image": os.path.join("src_train", name),
                "edited_image": os.path.join("dst_train", name),
                "image": [os.path.join("src_train", name), os.path.join("dst_train", name)],
            }
            data_list.append(data_dict)





#toy_qa = [{"name": "img_001.png", "gt_score": 4.43}] # ensure your GT score in range [0,5]

score2level = {5: "excellent", 4: "excellent", 3: "good", 2: "fair", 1: "poor", 0: "bad"}

conv_template = [
    {"from": "human", "value": "How would you rate the quality of this edited image? <|image|> <|image|>"},
    {"from": "gpt", "value": "The quality of the image is "}
]

toy_train_df = []

for toy_di in data_list:
    toy_train_di = {}
    # 直接复用已准备好的字段
    toy_train_di["src_image"] = toy_di["src_image"]
    toy_train_di["edited_image"] = toy_di["edited_image"]
    toy_train_di["image"] = toy_di["image"]
    toy_train_di["instruction"] = toy_di["instruction"]
    toy_train_di["gt_score"] = toy_di["gt_score"]
    # 会被 train_mem.py 在 dual_image_mode 下用 instruction 覆盖为 "{instruction} <|image|><|image|>"
    toy_train_di["conversations"] = copy.deepcopy(conv_template)
    # 构造监督信号（助手端）
    toy_train_di["conversations"][-1]["value"] = toy_train_di["conversations"][-1]["value"] + (score2level[math.floor(toy_di["gt_score"])]) + "."
    print(toy_train_di["gt_score"],toy_train_di["conversations"][-1]["value"])
    toy_train_df.append(toy_train_di)

#print(toy_train_df)

train_list=toy_train_df
# 将列表转换为JSON格式并写入文件
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(train_list,json_file, ensure_ascii=False, indent=4)


# test_list=toy_train_df[100:200]
# train_list=toy_train_df[:1400]+toy_train_df[2800:]
# # 将列表转换为JSON格式并写入文件
# with open('./t2vqa/train_split_2.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_list,json_file, ensure_ascii=False, indent=4)
# with open('./t2vqa/test_split_2.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_list,json_file, ensure_ascii=False, indent=4)
    

# test_list=toy_train_df[200:300]
# train_list=toy_train_df[:2800]+toy_train_df[4200:]

# # 将列表转换为JSON格式并写入文件
# with open('./t2vqa/train_split_3.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_list,json_file, ensure_ascii=False, indent=4)
# with open('./t2vqa/test_split_3.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_list,json_file, ensure_ascii=False, indent=4)
    
# test_list=toy_train_df[:5600]
# train_list=toy_train_df[:4200]+toy_train_df[5600:]

# # 将列表转换为JSON格式并写入文件
# with open('./t2vqa/train_split_4.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_list,json_file, ensure_ascii=False, indent=4)
# with open('./t2vqa/test_split_4.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_list,json_file, ensure_ascii=False, indent=4)
    
    
# test_list=toy_train_df[5600:]
# train_list=toy_train_df[:5600]

# # 将列表转换为JSON格式并写入文件
# with open('./t2vqa/train_split_5.json', 'w', encoding='utf-8') as json_file:
#     json.dump(train_list,json_file, ensure_ascii=False, indent=4)
# with open('./t2vqa/test_split_5.json', 'w', encoding='utf-8') as json_file:
#     json.dump(test_list,json_file, ensure_ascii=False, indent=4)
print(f"JSON文件已生成: {json_file_path}")
