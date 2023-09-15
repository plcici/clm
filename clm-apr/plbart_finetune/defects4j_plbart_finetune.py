import codecs
import json
import sys
import os
import re
import torch
import subprocess
from transformers import PLBartForConditionalGeneration, PLBartTokenizer

PLBART_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1] #返回当前脚本文件夹目录，os.path.abspath(__file__)获取当前脚本的绝对路径。rindex在给定字符串中，从右边开始查找/字符并返回它的索引值
JAVA_DIR = PLBART_FINETUNE_DIR + '../../jasper/'  #jasper：是一个解析Java文件并为不同代码语言模型准备输入的Java工具

def command(cmd):  
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) #Popen函数用于创建一个子进程，执行cmd命令，将结果保存在output和err中
    output, err = process.communicate()  #等待外部命令执行完成，返回一个元组(output, err)
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_plbart_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR) #更换目录到JAVA_DIR
    print(os.getcwd())
    #使用java命令，运行clm.finetuning.FineTuningData类，获取buggy_file的输入至defects4J/tmp.json文件中
    command([
        'java', '-cp', '/home/sunwanqi/caowy/APR/clm/jasper:/home/sunwanqi/caowy/APR/clm/jasper/target/classes:/home/sunwanqi/caowy/APR/clm/jasper/lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

#读取defects4j_loc.txt文件，获取对应模型的每个bug的输入将其写入plbart_input.json文件中
def defects4j_plbart_finetune_input(output_file, tmp_dir):
    loc_fp = codecs.open(PLBART_FINETUNE_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')  #codecs函数用于编码和解码，open函数打开文件，r表示读取，utf-8表示编码格式，在处理复杂的编解码任务时非常有用
    plbart_input = {'config': 'finetune', 'data': {}}
    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = PLBART_FINETUNE_DIR + '../defects4j/tmp.json'
        #使用defects4j checkout命令，将bug_idb版本的项目检出到tmp_dir目录下
        anstmp = subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(anstmp.stdout.decode('utf-8'))
        print(anstmp.stderr.decode('utf-8'))
        #
        get_plbart_finetune_input(tmp_dir + path, start, end, tmp_file)
        #TODO:输出err信息，但output为空
        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        if result["buggy function before"].strip() == '' and result["buggy line"].strip() == '' and result["buggy function after"].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue
        result = json.load(open(tmp_file, 'r'))
        buggy_function_before = re.sub('\\s+', ' ', result['buggy function before']).strip()  #\s匹配任何空白字符，+表示匹配前面的子表达式一次或多次，strip()函数用于去除字符串首尾的空格
        buggy_line = re.sub('\\s+', ' ', result['buggy line']).strip()
        buggy_function_after = re.sub('\\s+', ' ', result['buggy function after']).strip()
        inputs = '<s> ' + buggy_function_before + ' </s> ' + buggy_line + ' </s> ' + buggy_function_after + ' </s> java'
        plbart_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
            'loc': rem_loc,
            'input': inputs,
        }
        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(plbart_input, open(output_file, 'w'), indent=2)

def defects4j_plbart_finetune_output(input_file, output_file, model_dir, model_name, num_output=10):
    model = PLBartForConditionalGeneration.from_pretrained(model_dir + "/" + model_name)
    model = torch.nn.DataParallel(model, device_ids=device_id).to(device)
    tokenizer = PLBartTokenizer.from_pretrained(model_dir + "/" + model_name[:-9], src_lang="java", tgt_lang="java")  
    
    plbart_output = json.load(open(input_file, 'r'))
    plbart_output['model'] = model_name
    for filename in plbart_output['data']:  #filename: 'Chart_26_source/org/jfree/chart/axis/Axis.java_1192-1193'
        text = plbart_output['data'][filename]['input']   #获取输入

        print('generating', filename)
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        if input_ids.size(1) >= 512:  #过滤掉长度大于512的输入
            print('too long:', input_ids.size(1))
            continue

        input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        generated_ids = model.module.generate(  #当出现报错“DataParallel object has no attribute xxx”时，需要使用model.module.xxx
            input_ids, max_length=128, num_beams=num_output, num_return_sequences=num_output, 
            early_stopping=True, decoder_start_token_id=tokenizer.lang_code_to_id["__java__"]  #decoder_start_token_id表示decoder的起始token，对于某些模型，如sequence-to-sequence模型，这是非常关键的；表明解码器在开始时会使用一个特定的Java语言标记作为起始token
        ) #使用了DataParallel，原始模型会被封装在".module"中；max_length是生成文本的最大长度；num_beams表示beam search的数量；num_return_sequences表示返回的序列数量；early_stopping表示是否提前停止搜索
        output = []
        for generated_id in generated_ids:  #generated_ids是生成的10个候选项
            output.append(tokenizer.decode(generated_id, skip_special_tokens=True))
        plbart_output['data'][filename]['output'] = output
        json.dump(plbart_output, open(output_file, 'w'), indent=2) #indent=2表示缩进2个空格


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    device_id = [0, 1, 2]   # need one GPU with 12GB memory
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i), i)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    #model_dir = sys.argv[1]
    model_dir = "/home/sunwanqi/caowy/APR/clm/models"
    
    input_file = PLBART_FINETUNE_DIR + '../defects4j/plbart_finetune_result/plbart_input.json' #定义输入文件位置
    print("==========Preparing input of Defects4J benchmark to finetuned PLBART model==========")
    defects4j_plbart_finetune_input(input_file, tmp_dir='/tmp/plbart/')
    print("==========Input written to " + input_file)
    
    for model_name in ('plbart-base-finetune', 'plbart-large-finetune'):
        output_file = PLBART_FINETUNE_DIR + '../defects4j/plbart_finetune_result/' + '_'.join(model_name.split('-')[:-1]) + '_output.json'
        # model_dir = PLBART_FINETUNE_DIR + '../../models/'
        print("==========Generating output of Defects4J benchmark by " + model_name + "==========")
        defects4j_plbart_finetune_output(input_file, output_file, model_dir, model_name, num_output=10)
        print("==========Output written to " + output_file)   