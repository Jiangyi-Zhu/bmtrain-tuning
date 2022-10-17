# bmtrain-tuning debug
1. 问题描述
在训练分类数量较多的下游任务时，xlm-roberta-base模型训练开始或者过程中会报gradient overflow的错误。
出现问题的数据集：emoevent（8分类）、GoEmotions（28分类）
通过调整adapter的位置和adapter层数，emoevent已经不会报错，但GoEmotion会在最多几百步的时候报错
2. 已尝试的方法
1. Gradient clipping：没用
2. 调整learning rate：尝试过1e-3 ~ 1e-10，没有区别
3. 检查数据集：用huggingface、或者bmtrain roberta-base跑不会overflow，所以数据集没有问题
4. 设置warmup：设置10%的warmup，没有区别
5. 调整adapter位置：有用，解决了emoevent overflow的问题，使训练更稳定了。adapter现在放在了feedforward block的LayerNorm层后面
  不同任务adapter需要放的位置可能不一样。goemotion任务用roberta-base训练时adaper放在ffn后面才不会overflow
[图片]
6. 调整prompt层数、adapter层数：有用，解决了emoevent overflow的问题。但是即使将prompt层数增加到400，adapter增加到128，GoEmotions任务还会报错。但是会晚几百步报错。
  用roberta-base试验adapter层数增加时反而会overflow，怀疑adapter的设置有问题，但检查过加载模型和加载adapter的代码，没有发现问题
7. 调整adapter初始化：尝试了adapter论文中的初始化设置(中间层数8，64，256，standard deviation:1e-2,1e-7)，或自己调了不同的std，没有改善。
8. adapter的数据格式没有转换成fp16：统一完数据格式以后可以让网络多跑几个epoch

bmtrain-fp16 log文件：
1. xlm模型
    如果adapter layer 只有6层，prompt layer只有5层，能跑1个epoch
    其他layer组合（adapter:32,64,128,256, prompt:50,100,200,400）都会在几百步报错
2. roberta模型
    如果adapter layer 只有6层，prompt layer只有5层，能跑完20个epoch，loss为1.3
    如果adapter layer 有256层，prompt layer有400层， 能跑3个eopch，然后报错。但是loss为1.2
