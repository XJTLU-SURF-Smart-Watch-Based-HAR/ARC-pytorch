ARC要围绕model-development进行设计开发，核心是model，辅助是模型的输入以及模型的训练+检验。
自动化在于调整模型的输入（feature extraction, feature selection），后端一套走完。
训练集和测试集一定要分开来，在微调模型确认前就已经固定--->确定交叉验证K=10系数，直接分好不动，然后再在各类微调模型上跑--->
计算成本
甚至leave-one-subject-evaluation哈哈哈哈






