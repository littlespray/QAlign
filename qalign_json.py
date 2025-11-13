import json
import math
import os

# 假设txt文件的路径为'data.txt'
# txt_file_path = '/userhome/crave-db/shuffle_final_prompt_train.txt'
# json_file_path = '/userhome/crave-db/shuffle_final_prompt_train.json'

# txt_file_path = '/userhome/hunyuan_video2.txt'
# json_file_path = '/userhome/hunyuan-test.json'

# txt_file_path = './IE-R1-4K/train.txt'
# json_file_path = './IE-R1-train-traindst.json'

txt_file_path = './IE-R1-4K/val.txt'
json_file_path = './IE-R1-val-traindst.json'

# 初始化一个空列表来存储字典
data_list = []

# 读取txt文件
with open(txt_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行尾的换行符并分割字符串
        parts = line.strip().split('|')
        if len(parts) >= 3:
            name = parts[0]
            prompt = parts[3] # 第4个是dst prompt：{img_basename}|{instruct}|{src_caption}|{dst_caption}|{text_score}|{fidelity_score}|{quality_score}|{overall_score}
            label = parts[-1]
            # label=float(label) / 20 # T2V-QA
            # label=float(label)/2 # 如果是1-10分就需要除以2
            label = float(label)
            # 创建一个字典并添加到列表中
            data_dict = {
                "name": name,
                "gt_score": label
            }
            data_list.append(data_dict)





#toy_qa = [{"name": "img_001.png", "gt_score": 4.43}] # ensure your GT score in range [0,5]

score2level = {5: "excellent", 4: "excellent", 3: "good", 2: "fair", 1: "poor", 0: "bad"}

conv_template = [{"from": "human", "value": "How would you rate the quality of this video? <|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|>"}, {"from": "gpt", "value": "The quality of the video is"}]

toy_train_df = []

for toy_di in data_list:
    toy_train_di = {}
    toy_train_di["image"] = toy_di["name"]
    toy_train_di["gt_score"] = toy_di["gt_score"]
    toy_train_di["conversations"] = [{"from": "human", "value": "How would you rate the quality of this video? <|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|><|image|>"}, {"from": "gpt", "value": "The quality of the video is "}]
#     print(toy_train_di["conversations"])
#     print(score2level[math.floor(toy_di["gt_score"])])
    toy_train_di["conversations"][-1]["value"] = toy_train_di["conversations"][-1]["value"]+(score2level[math.floor(toy_di["gt_score"])])+"."
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
