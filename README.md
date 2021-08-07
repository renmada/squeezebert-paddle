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