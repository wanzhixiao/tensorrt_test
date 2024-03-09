# -*- coding: UTF-8 -*-
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict, Callable
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModel, AutoTokenizer, ErnieModel, LongformerModel
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import transformers
import os
import onnx
import onnxruntime
import numpy as np
import time
import tensorrt as trt
import subprocess
import os

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)


class Model(nn.Module):
    def __init__(self, pretrained_model_path):
        super(Model, self).__init__()
        self.pre_model = ErnieModel.from_pretrained(pretrained_model_path)
        
    def forward(self, input_ids: Tensor,attention_mask:Tensor):
        '''
        :param input_ids:
        :param attention_mask:
        :return:
        '''
        output = self.pre_model(input_ids, attention_mask, output_hidden_states=True)
        sequence_output = output.last_hidden_state
        cls = sequence_output[:, 0]
        return cls


def predict_data_process(sentences, tokenizer, DEVICE):
    encoded_input = tokenizer.batch_encode_plus(sentences,
                                                max_length=2048,
                                                padding='max_length',
                                                truncation=True)

    input_ids = encoded_input.get('input_ids')
    token_type_ids = encoded_input.get('token_type_ids')
    attention_mask = encoded_input.get('attention_mask')

    input_ids = torch.LongTensor(input_ids).to(DEVICE)
    token_type_ids = torch.LongTensor(token_type_ids).to(DEVICE)
    attention_mask = torch.LongTensor(attention_mask).to(DEVICE)

    return_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    return return_data

def trans2onnx(torch_model, onnx_model_path, onnx_model_sim_path, input_ids, attention_mask):
    # 转onnx
    # 执行shell命令，返回执行结果
    def execute_shell_command(command):
        try:
            result = subprocess.check_output(command, shell=True)
            print(command,' 执行成功！')
            return result.decode('utf-8')
        except subprocess.CalledProcessError as e:
            return "执行命令失败：" + str(e)

    command1 = 'rm {} -f'.format(onnx_model_path)
    command2 = 'rm {} -f'.format(onnx_model_sim_path)
    execute_shell_command(command1)
    execute_shell_command(command2)

    torch.onnx.export(torch_model, 
                    (input_ids, attention_mask), 
                    onnx_model_path,
                    opset_version=10,
                    input_names=['input_ids','attention_mask'], # 模型输入名称
                    output_names=['output'],                    # 模型输出名称
                    dynamic_axes={
                        'input_ids': {0: 'batch_size'},
                        'attention_mask':{0:'batch_size'},
                        'output': {0:'batch_size'}
                    }
                )  ## 第一维可变，第0维默认维batch

    # '简化onnx'
    #  tensorrt 不支持Int64, 需要转成int32
    command3 = 'python -m onnxsim {} {}'.format(onnx_model_path, onnx_model_sim_path)
    p =os.system(command3)
    print(p)

# onnx 模型验证
def onnx_model_predict(model_path, input_ids, attention_mask):

    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(model_path, providers=[('CUDAExecutionProvider', {'device_id': 0}),'CPUExecutionProvider'])

    # 计算ONNX输出
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
                ort_session.get_inputs()[1].name: to_numpy(attention_mask)}
    ort_outs = ort_session.run(['output'], ort_inputs)
    ort_outs_ = ort_outs[0]
    return ort_outs_

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def save_pytorch_model(model_path, pretrain_model_path):
    model = AutoModel.from_pretrained(pretrain_model_path)
    torch.save(model.state_dict(), model_path)

def connvert_pytorch_onnx(model_path, onnx_model_path, onnx_model_sim_path, pretrain_model_path):
    '''
    pytorch转onnx
    '''
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. model
    torch_model = Model(pretrain_model_path)
    torch_model.load_state_dict(torch.load(model_path, map_location=DEVICE), False)
    # ↓下面这一句一定要，否则onnx和torch输出不一样
    torch_model.eval()          
    torch_model.to(DEVICE)
    print('1. model load down!')

    # 2. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    sentences = ['你好', '我好']
    inputs = predict_data_process(sentences,tokenizer,DEVICE)
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    print('2. tokenizer down!', input_ids.shape, attention_mask.shape)

    # 3. pth转onnx
    trans2onnx(torch_model, onnx_model_path, onnx_model_sim_path, input_ids, attention_mask)
    print('3. convert pytorch to onnx down!')

    # 4. pth模型输出
    torch_out = torch_model(input_ids, attention_mask)
    print("torch_out.shape:", torch_out.shape)
    print("torch_out:", torch_out)

    # 5. onnx模型输出
    onnx_out = onnx_model_predict(onnx_model_sim_path, input_ids, attention_mask)
    print("ort_outs:", onnx_out.shape)
    print("ort_outs:", onnx_out)
    

# ---------------------------------------------------------------------------------------------
def convert_onnx_to_engine(onnx_file_path, engine_file_path=None, max_batch_size=10):
    '''
    onnx转trt模型
    '''
    logeer = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


    def build_engine():
        with trt.Builder(logeer) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, logeer) as parser, \
                trt.Runtime(logeer) as runtime:

            profile = builder.create_optimization_profile()
            # 输入的各列，最小、最优、最大值
            profile.set_shape("input_ids", (1, 2048), (10, 2048), (10, 2048))
            # profile.set_shape("token_type_ids", (1, 2048), (10, 2048), (10, 2048))
            profile.set_shape("attention_mask", (1, 2048), (10, 2048), (10, 2048))

            config.add_optimization_profile(profile)
            config.max_workspace_size = 1 << 32
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096*(1 << 30))

            builder.max_batch_size = max_batch_size
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))

            print("Parsing ONNX file")
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            print('Completed parsing of ONNX file')

            print("Building TensorRT engine, This may take a few minutes.")
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(logeer) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


if __name__ == '__main__':
    # hugging face模型
    pretrain_model_path = './model_dir/ernie-3.0-base-zh/'
    # torch模型
    pytorch_file = "./model_dir/ernie_embedding.pth"
    # onnx int64模型
    onnx_file = "./model_dir/text_encoder1.onnx"
    # onnx int 32模型
    onnx_sim_file = "./model_dir/text_encoder_sim.onnx"
    # tensorrt 模型
    engine_file = "./model_dir/text_encoder1.trt"

    # 1. 保存pytorch模型
    save_pytorch_model(pytorch_file, pretrain_model_path)
    # 2. pytorch 转onnx
    connvert_pytorch_onnx(pytorch_file, onnx_file, onnx_sim_file, pretrain_model_path)
    # 3. onnx转engine
    convert_onnx_to_engine(onnx_sim_file, engine_file)
