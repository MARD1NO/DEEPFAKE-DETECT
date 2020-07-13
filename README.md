# DEEPFAKE-DETECT

# 7.13 UPDATE
1. 在后续的模型加入了Label Smooth功能
2. 更合理的数据增广，在同一视频不同帧所作增广相同。取消了验证集的增广，只做了归一化
3. 加入了GRU单元
4. 加入了MogGRU 和 MogLSTM单元（正在测试
5. 重新造了数据集，选的是DFDC中的part14分包数据
6. 更合理的训练集验证集分配策略，两者互不相交
7. FAKE:REAL比例在6.5:3.5
