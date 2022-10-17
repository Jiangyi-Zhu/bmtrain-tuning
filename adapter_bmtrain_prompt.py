#from openprompt.plms import load_plm
import time
from openprompt import PromptDataLoader
import torch
from data_process import get_dataset
from source_code_edited.manual_verbalizer_bmtrain import ManualVerbalizer
from source_code_edited.mixed_template_bmtrain import MixedTemplate
# from source_code_edited.module_bmtrain import PromptForClassification
# from source_code_edited.module_bmtrain import load_plm
# from source_code_edited.mixed_template_huggingface import MixedTemplate
from openprompt.prompts import ManualTemplate
from util import get_current_time, Logger, get_args
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from opendelta import Visualization
from opendelta import AdapterModel
import sys
import os
import bmtrain as bmt

def prepare(dataset, labels, args):
    if args.model_class in ['bert-base-cased','bert-large-cased']: model = 'bert'
    elif args.model_class in ['roberta-base','roberta-large']: model = 'roberta'
    elif args.model_class in ['xlm-roberta-base','xlm-roberta-large']: model = 'xlm'
    elif args.model_class in ['vinai/bertweet-base']: model = 'bertweet'
    elif args.model_class in ['cardiffnlp/twitter-xlm-roberta-base']: model = 'twitter-xlm'
    elif args.model_class in ['hfl/chinese-roberta-wwm-ext','hfl/chinese-roberta-wwm-ext-large']: model = 'chinese-roberta'
    elif args.model_class in ['weibo-bert-base','weibo-bert-large']: model = 'weibo-bert'
    
    if args.source == 'bmtrain':
        plm, tokenizer, model_config, WrapperClass = load_plm(model, args.model_class, dtype=args.dtype)
    elif args.source == "huggingface":
        plm, tokenizer, model_config, WrapperClass = load_plm(model, args.model_class)

    # 创建prompt template
    if args.language == 'English':
        if args.soft_layer == 0:
            mytemplate = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} It was {"mask"}')
        else:
            template_text = '{"placeholder":"text_a"} It was {"mask"} {"soft": None, "duplicate":' + str(args.soft_layer) + '}'
            print('template_text:',template_text)
            mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    if args.language == 'Chinese':
        if args.soft_layer == 0:
            mytemplate = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} 这句话的评价是 {"mask"}')
        else:
            template_text = '{"placeholder":"text_a"} 这句话的态度是 {"mask"} {"soft": None, "duplicate":' + str(args.soft_layer) + '}'
            print('template_text:',template_text)
            mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)
    myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(labels),label_words=labels)
    if model =='bertweet':
        max_seq_length = 128
    elif model =='chinese-roberta':
        max_seq_length = 500
    else:
        max_seq_length = 512
    
    # 转换数据集
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, decoder_max_length=3,
        batch_size=args.batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    valid_dataloader = PromptDataLoader(dataset=dataset["valid"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, decoder_max_length=3,
        batch_size=args.batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_length, decoder_max_length=3,
        batch_size=args.batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    print('len(train_dataloader)',len(train_dataloader))
    print('len(valid_dataloader)',len(valid_dataloader))
    print('len(test_dataloader)',len(test_dataloader))
    print("train data truncate rate: {}".format(train_dataloader.tokenizer_wrapper.truncate_rate), flush=True)
    
    # 添加adapter
    if args.adapter_bottleneck_dim != 0:
        # 'attention.output.LayerNorm'
        if args.source == 'bmtrain':
            #ffn.ffn   ffn.layernorm_before_ffn
            delta_model = AdapterModel(backbone_model=plm, modified_modules=['ffn.ffn'], bottleneck_dim=args.adapter_bottleneck_dim)
        elif args.source =='huggingface':
            delta_model = AdapterModel(backbone_model=plm, modified_modules=['attention.output.LayerNorm'], bottleneck_dim=args.adapter_bottleneck_dim)
        delta_model.freeze_module(exclude=["adapter"], set_state_dict=True)
    else:delta_model=None

    # 添加prompt
    prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
    # 从checkpoint加载prompt
    if args.soft_template_load_path != 'None':
        prompt_model.template.soft_embedding.weight.data = torch.load(args.soft_template_load_path)
    
    # adapter 自定义初始化
    if args.adapter_init_std != 0:
        for name, param in delta_model.named_parameters():        
            if 'weight' in name:
                print(name,'   ',param.shape,'   ','require_grad:',param.requires_grad)
                print('before:',param)
                torch.nn.init.normal_(param,mean=0.0, std=args.adapter_init_std)
                print('after:',param)
    # 从checkpoint加载adapter
    if args.adapter_load_path != 'None':
        delta_model.load_state_dict(torch.load(args.adapter_load_path))
    #bmt.init_parameters(prompt_model)
    Visualization(prompt_model).structure_graph()

    # 统一网络层数据格式
    if args.dtype == 'float16': 
        for name, param in prompt_model.named_parameters():
            if param.dtype == torch.float32:
                param=param.half()
            if args.pattern =='debug':
                print(name,"\n    ",param.shape,param.dtype,'   ','require_grad:',param.requires_grad)
    elif args.dtype == 'float32': 
        for name, param in prompt_model.named_parameters():
            if param.dtype == torch.float16:
                param=param.float()
            if args.pattern =='debug':
                print(name,"\n    ",param.shape,param.dtype,'   ','require_grad:',param.requires_grad)

    return prompt_model,delta_model,train_dataloader,valid_dataloader,test_dataloader

def train(prompt_model,delta_model,train_dataloader,valid_dataloader,args,
        device=0,checkpoint_save_path=''):
    '''训练模型
    '''
    import torch
    torch.cuda.empty_cache()
    fix_seed = 20
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    use_cuda = True
    torch.cuda.empty_cache()
    if use_cuda:
        prompt_model =  prompt_model.cuda(device)

    steps = len(train_dataloader)
    total_steps = steps*args.epochs
    optimizer_grouped_parameters = [{'params': [p for n,p in prompt_model.named_parameters() if "soft_embedding" in n or "adapter" in n]}]
    if args.dtype == 'float32':
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0*total_steps, num_training_steps=total_steps) # usually num_warmup_steps is 500
    elif args.dtype == 'float16':
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100) # if half trainning
        optimizer = bmt.optim.AdamOffloadOptimizer(optimizer_grouped_parameters, weight_decay=0)
        # loss_func = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        if args.lr_scheduler == 'NoDecay':
            scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                    start_lr = args.learning_rate,
                                    warmup_iter = 0*total_steps, 
                                    end_iter=total_steps,
                                    num_iter = 1)
        elif args.lr_scheduler == 'Linear':
            scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                    start_lr = args.learning_rate,
                                    warmup_iter = 0*total_steps, 
                                    end_iter=total_steps,
                                    num_iter = 1)

    best_prompt_model = prompt_model
    lowest_loss = float('inf')
    #torch.autograd.set_detect_anomaly(True)

    # 开始训练
    for epoch in range(args.epochs):
        if args.pattern == "train":
            prompt_model.train()
        elif args.pattern =='debug':
            prompt_model.eval()
        print(f'\n================ Epoch {epoch} ================')
        if args.dtype == 'float32': print('learning rate:',scheduler.get_last_lr())
        elif args.dtype == 'float16': print('learning rate:',scheduler.get_lr())
        tot_loss = 0
        epoch_start_time = time.time()
        st_time = time.time()
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda(device)
            logits = prompt_model(inputs)
            #logits = torch.nan_to_num(logits, neginf=0)
            labels = inputs['label']
            loss = loss_func(logits, labels)

            if args.pattern =='debug':
                print("loss:",loss)
                print('===============logits===============')
                print(logits)
                print('===============label===============')
                print(labels)

            del inputs
            torch.cuda.empty_cache()

            if args.dtype == 'float32':
                loss.backward()
                if args.pattern =='debug':
                    for i,j in prompt_model.named_parameters():
                        if 'soft_embedding' in i:
                            print(i)
                            print("=============prompt.data 1===========")
                            print(j.weight.data.clone())
                            print("=============prompt.grad 1===========")
                            print(j.grad.data.clone())
                optimizer.step()
                scheduler.step()

            elif args.dtype == 'float16':
                loss = optimizer.loss_scale(loss)
                loss.backward()
                if args.pattern =='debug':
                    print('step:',step)
                    # for i,j in prompt_model.named_parameters():
                    #     if j.grad != None:
                    #         print('==================')
                    #         print(j.data.sum())
                    #         print(j.grad.data.sum())
                    #         print(j.grad.data)
                    #         j.grad.data=j.grad.data/8
                    #         print(j.grad.data)
                    # print('----------------------------------------')
                    # for i in optimizer_grouped_parameters:
                    #     print(i)
                    #     print(i.grad)
                    # print(optimizer_grouped_parameters)
                    # print('----------------------------------------')
                    #print(optimizer_grouped_parameters)
                    #print(optimizer_grouped_parameters[0]['params'][0].weight.data)
                    for i,j in prompt_model.named_parameters():
                        if 'prompt_model.template' in i:
                            print(i)
                            print("=============prompt.data 1===========")
                            print(j.data.clone())
                            print("=============prompt.grad 1===========")
                            print(j.grad.data.clone())
                    for i,j in prompt_model.named_parameters():
                        if 'layers.0.ffn.layernorm_before_ffn.adapter' in i:
                            print(i)
                            print("=============adapter.data 1===========")
                            print(j.data.clone())
                            print("=============adapter.grad.data 1===========")
                            print(j.grad.data.clone())
                bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm= 1, scale = optimizer.scale, norm_type =  2)
                # optimizer.step()
                # scheduler.step()
                bmt.optim_step(optimizer,scheduler)
            
                # if args.pattern =='debug':
                    # for i,j in prompt_model.named_parameters():
                    #         if 'prompt_model.template' in i:
                    #             print(i)
                    #             print("=============prompt.data 2===========")
                    #             print(j.data)
                    #             print("=============prompt.grad 2===========")
                    #             print(j.grad.data)
                    # print('----------------------------------------')
                    # for i in optimizer_grouped_parameters.params:
                    #     print(i)
                    #     print(i.grad)
                    # print(optimizer_grouped_parameters)
                    # print('----------------------------------------')
            tot_loss += loss.item()
            optimizer.zero_grad()
            
            if step in [0,int(steps*0.01),int(steps*0.02),int(steps*0.03),int(steps*0.04),int(steps*0.05),int(steps*0.3),int(steps*0.5),int(steps*0.75)]:
                elapsed_time = time.time() - st_time
                st_time = time.time()
                print(f'Progress {round((step+1)/steps*100,2)}%, loss: {tot_loss/(step+1)}, time {round(elapsed_time,2)}s')
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f'Epoch {epoch}, average train loss: {tot_loss/(step+1)}, time {round(epoch_elapsed_time,2)}s')
        tot_loss = 0

        # 用validation数据集验证中间结果
        allpreds = []
        alllabels = []
        prompt_model.eval()
        for step, inputs in enumerate(valid_dataloader):
            if use_cuda:
                inputs = inputs.cuda(device)
            logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            del inputs
            torch.cuda.empty_cache()
            tot_loss += loss.item()
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        print("Epoch {}, average test loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

        # save checkpoint
        acc, F1, precision, recall = compute_metrics(alllabels, allpreds)
        if tot_loss<lowest_loss:
            lowest_loss = tot_loss
            print('lowest_loss updated')
            best_prompt_model = prompt_model
            propmt_path = f'{checkpoint_save_path}/lowest_loss_soft_template__epoch-{epoch}__acc-{str(acc*100)[:5]}__F1-{str(F1*100)[:5]}_prompt.pt'
            adapter_path = f'{checkpoint_save_path}/lowest_loss_soft_template__epoch-{epoch}__acc-{str(acc*100)[:5]}__F1-{str(F1*100)[:5]}_adapter.pt'
            if args.soft_layer != 0:
                torch.save(prompt_model.template.soft_embedding.weight.data.cpu(),propmt_path)
            if args.adapter_bottleneck_dim != 0:
                torch.save(delta_model.state_dict(), adapter_path)
    return best_prompt_model

def inference(prompt_model,test_dataloader,device=0):
    print(f'\n================ inference on test dataset ================')
    prompt_model.eval()
    use_cuda = True
    if use_cuda:
        prompt_model =  prompt_model.cuda(device)
    torch.cuda.empty_cache()
    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda(device)
        logits = prompt_model(inputs)
        labels = inputs['label']
        del inputs
        torch.cuda.empty_cache()
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc, F1, precision, recall = compute_metrics(alllabels, allpreds)
    #acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    torch.cuda.empty_cache()

def compute_metrics(alllabels, allpreds):
    precision, recall, F1, _ = precision_recall_fscore_support(alllabels, allpreds, average='macro')
    acc = accuracy_score(alllabels, allpreds)
    print('acc:',acc,'\nF1:',F1,'\nprecision:',precision,'\nrecall:',recall)
    return acc, F1, precision, recall

if __name__=='__main__':
    args = get_args()
    bmt.init_distributed(seed=args.seed, loss_scale_factor = 2, loss_scale_steps = 100)
    if args.source == 'bmtrain':
        from source_code_edited.module_bmtrain import PromptForClassification
        from source_code_edited.module_bmtrain import load_plm
        from source_code_edited.mixed_template_bmtrain import MixedTemplate
    elif args.source == 'huggingface':
        #from source_code_edited.module_huggingface import PromptForClassification
        from openprompt import PromptForClassification
        from openprompt.plms import load_plm
        #from source_code_edited.module_huggingface import load_plm
        from source_code_edited.mixed_template_huggingface import MixedTemplate
    dataset,labels = get_dataset(args.task)   
    checkpoint_save_path = f'{args.checkpoint_save_path}/adapter/{args.task}/{args.model_class}/{get_current_time()}_{args.source}_{args.dtype}_seed_{args.seed}'
    if args.pattern == 'train' or args.pattern == 'debug': # 如果模式为train，创建一个checkpoint文件夹
        if not os.path.exists(checkpoint_save_path): 
            os.makedirs(checkpoint_save_path)
        sys.stdout = Logger(checkpoint_save_path) # 把print输出到logger
        print('checkpoint_save_path:',checkpoint_save_path)
    print(args)
    
    prompt_model,delta_model,train_dataloader,valid_dataloader,test_dataloader = prepare(dataset,labels,args)
    if args.pattern == 'train' or args.pattern == 'debug':
        prompt_model = train(prompt_model,delta_model,train_dataloader,valid_dataloader,args,
        device=0, checkpoint_save_path=checkpoint_save_path)
    inference(prompt_model,test_dataloader,device=0)