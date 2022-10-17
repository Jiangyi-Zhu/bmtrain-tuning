device=5
pattern='debug' # train inference debug
source='bmtrain' # bmtrain huggingface
# sentiment: SemEval4a twitter_US_airline first_GOP trump_tweet sentiment140 SST-2 SST-2-openprompt CoLA HCR
# 中文： nlpcc2012 shanghai_task3 douban_book_review
# emotion: go_emotions emoevent emoint goodnewseveryone semeval2018task1 
# 中文： nlpcc2013 nlpcc2014 nlpcc2018 covid_19_sentiment smp2020_ewect_usual smp2020_ewect_virus
# offensive: OLID
tasks="go_emotions"
# roberta-base roberta-large vinai/bertweet-base
# xlm-roberta-base xlm-roberta-large cardiffnlp/twitter-xlm-roberta-base
# 中文模型： hfl/chinese-roberta-wwm-ext hfl/chinese-roberta-wwm-ext-large weibo-bert-base weibo-bert-large
model_classes="xlm-roberta-base"
language="English" # Chinese English
soft_layer=5 # 5 10
adapter_bottleneck_dim=6 #6 10
adapter_init_std=0 # if use defult, here is 0, std:0.01
epochs=20 # 20 40
learning_rate=1e-4
batch_size=64 # 64
seeds="78" # 20 42 78, (100)
dtype='float16' # float32 float16
lr_scheduler='Linear' # NoDecay, Linear
soft_template_load_path='None' # 不读这里写'None'
adapter_load_path='None' # 不读这里写'None'
checkpoint_save_path='/home/zhujiangyi/prompt_learning/checkpoints'

for task in ${tasks}; do {
    for seed in ${seeds}; do {
        for model_class in ${model_classes}; do {
            CUDA_VISIBLE_DEVICES=${device} \
            torchrun  --master_port=5508${device} \
            adapter_bmtrain_prompt.py \
                --pattern ${pattern} \
                --source ${source} \
                --task ${task} \
                --model_class ${model_class} \
                --language ${language} \
                --soft_layer ${soft_layer} \
                --adapter_bottleneck_dim ${adapter_bottleneck_dim} \
                --adapter_init_std ${adapter_init_std} \
                --epochs ${epochs} \
                --learning_rate ${learning_rate} \
                --batch_size ${batch_size} \
                --seed ${seed} \
                --dtype ${dtype} \
                --lr_scheduler ${lr_scheduler} \
                --soft_template_load_path ${soft_template_load_path} \
                --adapter_load_path ${adapter_load_path} \
                --checkpoint_save_path ${checkpoint_save_path}
        }
        done
    }
    done
}
done