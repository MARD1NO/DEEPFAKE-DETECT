# DEEPFAKE-DETECT

# 7.13 UPDATE
1. 在后续的模型加入了Label Smooth功能
2. 更合理的数据增广，在同一视频不同帧所作增广相同。取消了验证集的增广，只做了归一化
3. 加入了GRU单元
4. 加入了MogGRU 和 MogLSTM单元（正在测试
5. 重新造了数据集，选的是DFDC中的part14分包数据
6. 更合理的训练集验证集分配策略，两者互不相交
7. FAKE:REAL比例在6.5:3.5

# 7.15 FIX BUG

1. 修复了EfficientNet跨层连接bug，现在跨层连接与drop connect运行正常
2. 为后续模型增加了一键dropout，方便evaluate
3. 修复了ConvLSTM 的bug，将隐层正确初始化
4. 修复了eval验证时，输出不相同的bug（原因出自LSTM单元初始化参数没有加入到动态图当中)

# 7.23

1. 增加了新的数据截取方式，将每一个FAKE视频对应的REAL 视频作为一对，分别裁剪出人脸，放入一组
2. 新的DataLoader，将每一组含有FAKE和REAL的数据，整合进一个batch