# -*- coding: UTF-8 -*-
'''
tensorrt模型推理
'''
import torch
import argparse
import numpy as np
import tensorrt as trt
import common
import time
from transformers import BertTokenizer

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def predict_data_process(sentences, tokenizer, device):
    encoded_input = tokenizer.batch_encode_plus(sentences,
                                                max_length=2048,
                                                padding='max_length',
                                                truncation=True)

    input_ids = encoded_input.get('input_ids')
    token_type_ids = encoded_input.get('token_type_ids')
    attention_mask = encoded_input.get('attention_mask')

    input_ids = torch.LongTensor(input_ids).to(device)
    token_type_ids = torch.LongTensor(token_type_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    return_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    return return_data

def load_model(args):
    tokenizer = BertTokenizer(vocab_file=args.bert_model + "/vocab.txt")
    engine = get_engine(args.trt_model)
    return engine, tokenizer


def create_context(input_shape, engine):

    time1 = time.time()
    context = engine.create_execution_context()
    print('1. create_execution_context:',time.time()-time1, context)

    """
    b.从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
    """
    time1 = time.time()
    context.active_optimization_profile = 0
    origin_inputshape = context.get_binding_shape(0)  # (1,-1)
    print('2. origin_inputshape = ',origin_inputshape)

    origin_inputshape[0], origin_inputshape[1] = input_shape  # (batch_size, max_sequence_length)
    print('3. origin_inputshape[0], origin_inputshape[1] = ', origin_inputshape[0], origin_inputshape[1])
    # 模型输入是[input_ids, token_type_ids, attention_mask]这三列，如果输入特征列只有两列，去掉一列。
    context.set_binding_shape(0, (origin_inputshape))
    context.set_binding_shape(1, (origin_inputshape))
    # context.set_binding_shape(2, (origin_inputshape))
    print('4. set_bindings:',time.time()-time1)

    return context


def infer2(input_ids, attention_mask, engine, context, device):

    # tokenizer = BertTokenizer(vocab_file=args.bert_model + "/vocab.txt")

    """
    a.获取engine, 建立上下文
    """

    """
    c、输入数据填充
    """
    time1 = time.time()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine, input_ids.shape[0])
    inputs[0].host = input_ids
    inputs[1].host = attention_mask
    # inputs[2].host = attention_mask
    # print('1. allocate_buffers:',time.time()-time1) # 0.00077
    """
    d、tensorrt推理
    """
    time1 = time.time()

    print('input_ids:',input_ids.shape,attention_mask.shape, type(input_ids), input_ids.dtype)
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=input_ids.shape[0])
    # print('7. do_inference_v2',time.time()-time1)  #  0.469

    time1 = time.time()
    trt_outputs = trt_outputs[0].reshape(input_ids.shape[0], -1)  #
    # ort_outs_logits = torch.tensor(trt_outputs)
    # _, pred = torch.max(ort_outs_logits.data, 1)
    # 这里输出的是模型预测标签的ID
    # print('trt_outputs.shape = {}, used_time = {}'.format(trt_outputs.shape,time.time()-time1))
    return trt_outputs


if __name__ == '__main__':
    engine_file = "./model_dir/text_encoder1.trt"
    pretrain_model_path = './model_dir/ernie-3.0-base-zh/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_model', default=engine_file)
    parser.add_argument('--bert_model', default=pretrain_model_path)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    # sentences = ['你好', "我不好"]

    # 1. 加载数据
    import json
    with open("./text_data_array100.json", "r", encoding="utf-8") as f:
        text = json.load(f)

    # text = ['你好', '我好']

    # 2. 加载模型
    time1 = time.time()
    engine, tokenizer = load_model(args)
    print('model load time:',time.time()-time1)

    # tokenizer
    time1 = time.time()
    inputs = predict_data_process(text, tokenizer, device)
    input_ids = to_numpy(inputs['input_ids'].int())
    attention_mask = to_numpy(inputs['attention_mask'].int())
    print('tokenizer time:',time.time()-time1)

    time1 = time.time()
    n = len(text)
    step = 10
    output = []
    input_shape = (step, 2048)
    context = create_context(input_shape, engine)
    print('create_context time:',time.time()-time1)

    print(' engine = ', engine, type(engine))
    print(' context = ', context, type(context))

    time1 = time.time()
    for i in range(0,n,step):
        start_time = time.time()
        start = i
        end = min(i+step, n)
        # sentences = text[start:end]
        ids = input_ids[start:end]
        mask = attention_mask[start:end]
        trt_output = infer2(ids, mask, engine, context, device)
        end_time = time.time()
        print('size:{},batch time:{}'.format(len(ids),end_time-start_time))
        output.append(trt_output)

    out = np.concatenate(output,axis=0)
    time2 = time.time()
    print(out)
    print('used time:',time2-time1,' out.shape:',out.shape)

