# squeezebert-paddle

#当前成果
1. paddle版本的代码
2. pytorch转化的权重

# 前向传播精度和速度对比
执行这个文件 modeling.py，速度是在i5-7500上的表现  
 
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

# QQP数据集合效果  
eval loss: 0.727248, 
acc: 0.9074449666089538, 
precision: 0.8636037329504667, 
recall: 0.8890157877057441, 
f1: 0.876125529661017, 
acc and f1: 0.8917852481349854, 
eval done total : 196.50919675827026 s

运行参数
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