# squeezebert-paddle
## 验收标准
1. 完成模型权重从pytorch到paddle的转换代码，转换3个预训练权重（“squeezebert/squeezebert-uncased”，
“squeezebert/squeezebert-mnli”，“squeezebert/squeezebert-mnli-headless”）
2. "squeezebert/squeezebert-mnli-headless"模型指标：QQP验证集accuracy=89.4（见论文Table 2）
3. SqueezeBERT模型加速比对比BERT-Base达到4.3x（见论文Table 2）
4. 提交PR至PaddleNLP

## 权重下载
链接: https://pan.baidu.com/s/1Jis7In0veo4ODae5OR_FqA 提取码: p5bk

## 前向传播精度和速度对比
python compare.py

model_name: squeezebert-uncased  
min difference: 0.0   
max difference: 6.556511e-07  
squeeze cost 51.86033248901367, bert cost 83.77599716186523   

model_name: squeezebert-mnli   
min difference: 0.0   
max difference: 7.4505806e-07   


model_name: squeezebert-mnli-headless   
min difference: 0.0   
max difference: 7.4505806e-07 

## QQP数据集合效果 

```
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=QQP
python -u ./run_glue.py \
    --model_type squeezebert \
    --model_name_or_path ./ \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 16   \
    --learning_rate 3e-05 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 20000 \
    --output_dir ./tmp/$TASK_NAME/ \
    --device gpu
```
*运行结果*

| acc | precision | recall | f1 |
| :----:| :----:| :----: | :----:|
| 0.9074 | 0.8636 | 0.8890 | 0.8761 |
