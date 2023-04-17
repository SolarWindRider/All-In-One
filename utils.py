import configuration as conf
import requests
import tiktoken
import json
import threading
import random
from PIL import Image, ImageDraw
from diffusers.utils import load_image
import time
from huggingface_hub import InferenceApi
import io
import base64

# 喂给模型的prompt的组合方式是  hprompt+demonstration+用户实际输入+fprompt
hprompt = dict(
    parse_task="""#1 Task Planning Stage: The AI assistant can parse user input to several tasks: [{"task": task, "id": task_id, "dep": dependency_task_id, "args": {"text": text or <GENERATED>-dep_id, "image": image_url or <GENERATED>-dep_id, "audio": audio_url or <GENERATED>-dep_id}}]. The special tag "<GENERATED>-dep_id" refer to the one generated text/image/audio in the dependency task (Please consider whether the dependency task generates resources of this type.) and "dep_id" must be in "dep" list. The "dep" field denotes the ids of the previous prerequisite tasks which generate a new resource that the current task relies on. The "args" field must in ["text", "image", "audio"], nothing else. The task MUST be selected from the following options: "token-classification", "text2text-generation", "summarization", "translation", "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "text-to-video", "visual-question-answering", "document-question-answering", "image-segmentation", "depth-estimation", "text-to-speech", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "canny-control", "hed-control", "mlsd-control", "normal-control", "openpose-control", "canny-text-to-image", "depth-text-to-image", "hed-text-to-image", "mlsd-text-to-image", "normal-text-to-image", "openpose-text-to-image", "seg-text-to-image". There may be multiple tasks of the same type. Think step by step about all the tasks needed to resolve the user's request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If the user input can't be parsed, you need to reply empty JSON [].""",
    # choose_model="""#2 Model Selection Stage: Given the user request and the parsed tasks, the AI assistant helps the user to select a suitable model from a list of models to process the user request. The assistant should focus more on the description of the model and find the model that has the most potential to solve requests and tasks. Also, prefer models with local inference endpoints for speed and stability.""",
    response_results="""# 4 Response Generation Stage: With the task execution logs, the AI assistant needs to describe the process and inference results.""")

demonstration = dict(
    parse_task=json.load(open("./demos/parse_task.json", "r", encoding="utf-8")),
    # choose_model=json.load(open("./demos/choose_model.json", "r", encoding="utf-8")),
    response_results=json.load(open("./demos/response_results.json", "r", encoding="utf-8")),
)

fprompt = dict(
    parse_task="""The chat log [ {{context}} ] may contain the resources I mentioned. Now I input { {{userinput}} }. Pay attention to the input and output types of tasks and the dependencies between tasks.""",
    # choose_model="""Please choose the most suitable model from {{metas}} for the task {{task}}. The output must be in a strict JSON format: {"id": "id", "reason": "your detail reasons for the choice"}.""",
    response_results="""Yes. Please first think carefully and directly answer my request based on the inference results. Some of the inferences may not always turn out to be correct and require you to make careful consideration in making decisions. Then please detail your workflow including the used models and inference results for my request in your friendly tone. Please filter out information that is not relevant to my request. Tell me the complete path or urls of files in inference results. If there is nothing in the results, please tell me you can't make it. }"""
)

MODELS_MAP = {}
for model in [json.loads(line) for line in open("./modelmap.jsonl", "r", encoding="utf-8").readlines()]:
    tag = model["task"]
    if tag not in MODELS_MAP:
        MODELS_MAP[tag] = []
    MODELS_MAP[tag].append(model)

HUGGINGFACE_HEADERS = {"Authorization": f"Bearer {conf.HuggingfaceKey}"}


def chat(message, userinput):  # 当前版本让用户发送的每个任务之间没有联系
    output1 = step1(message)
    output2 = step2(output1)
    # output3 = step3(output2)# step2直接把任务分发给step3
    output4 = step4(userinput, output2)
    return output4


def step1(userinput):  # 任务分发
    # 判断有没有任务
    if conf.TaskTrigger not in userinput[-1]["content"]:
        pass  # 没有任务的情况先不处理
    else:  # 任务拆分
        userinput[0]["content"] = userinput[-1]["content"].replace(conf.TaskTrigger, "")
        outputstr = chatGPTrequest(userinput)
        try:
            subtasks = json.loads(outputstr)
        except Exception as e:
            print(f"step1出错，无法解析出json:{e}")
    return subtasks


def step2(subtasks):  # 我这里假定了subtask的引用是完全正确的（单向无环图），然后把图压栈执行。
    queue = []
    while len(subtasks):
        max_dep, max_dep_id = -1, 0
        for _idx, task in enumerate(subtasks):
            max_dep__ = max(task["dep"])
            if max_dep < max_dep__:
                max_dep = max_dep__
                max_dep_id = _idx
        queue.append(subtasks.pop(max_dep_id))
    queue.reverse()  # 用列表模拟栈，所以要把1顺序转置一下

    # 不依赖其他任务执行结果的任务首先用多线程跑结果，然后其他任务顺序执行。
    results = dict()
    threads, seq = [], []
    for task in queue:
        if task["dep"] == [-1]:
            thread = threading.Thread(target=step3, args=(task, results))
            thread.start()
            threads.append(thread)
        else:
            seq.append(task)
    for thread in threads:
        thread.join()

    # 依赖其他任务执行结果的按照顺序执行即可
    for task in seq:
        for k in task["args"].keys():
            if "GENERATED" in task["args"][k]:
                task["args"][k] = \
                    str(results[int(task["args"][k].split("-")[-1])]["inference result"].values()).split("\'")[1]

        step3(task, results)
    return results


def step3(taskjson, results):  # 收集专家模型的处理结果， 目前只能文本、图像、语音三个模态。
    # Text only
    if taskjson["task"] in ["summarization", "translation", "conversational", "text-generation",
                            "text2text-generation", "text2text-generation"]:
        best_model_id = "ChatGPT"
        reason = "ChatGPT performs well on some NLP tasks as well."
        choose = {"id": best_model_id, "reason": reason}
        messages = [{
            "role": "user",
            "content": f"[ {taskjson} ] contains a task in JSON format {taskjson}. Now you are a {taskjson['task']} system, the arguments are {taskjson['args']}. Just help me do {taskjson['task']} and give me the result. The result must be in text form without any urls."
        }]
        inference_result = {"generated text": chatGPTrequest(messages, isStep1=False)}


    else:  # 图像和语音问题需要用到huggingface的专家模型
        model_id, choose_reason = chooes_best_model(taskjson["task"])
        choose = {"id": model_id, "reason": choose_reason}
        inference = InferenceApi(repo_id=model_id, token=conf.HuggingfaceKey)

        # Image only
        if taskjson["task"] in ["object-detection", "image-to-image", "image-segmentation", ]:
            if taskjson["task"] == "object-detection":
                inference_result = object_detection(taskjson, inference)
            elif taskjson["task"] == "image-segmentation":
                inference_result = image_segmentation(taskjson, inference)
            elif taskjson["task"] == "image-to-image":
                inference_result = image_to_image(taskjson, model_id)



        # Audio only
        elif taskjson["task"] in ["audio-to-audio", ]:
            pass

        # Text & Image
        elif taskjson["task"] in ["image-to-text", "text-to-image", "visual-question-answering",
                                  "document-question-answering", "image-classification"]:
            if taskjson["task"] == "image-to-text":
                inference_result = image_to_text(taskjson, model_id)
            elif taskjson["task"] == "text-to-image":
                inference_result = text_to_image(taskjson, inference)
            elif taskjson["task"] == "visual-question-answering" or taskjson["task"] == "document-question-answering":
                model_id = "dandelin/vilt-b32-finetuned-vqa"
                inference_result = visual_question_answering(taskjson, model_id)


        # Text & Audio
        elif taskjson["task"] in ["text-to-speech", "automatic-speech-recognition", "audio-classification"]:
            if taskjson["task"] == "text-to-speech":
                inference = InferenceApi(repo_id='espnet/kan-bayashi_ljspeech_vits', token=conf.HuggingfaceKey)
                inference_result = text_to_speech(taskjson, inference)

        else:
            print("任务过于复杂，暂无法处理")
    results[taskjson["id"]] = collect_result(taskjson, choose, inference_result)
    results = results.copy()
    return results


def step4(userinput, results):  # 汇总处理结果，回答用户
    results = [v for k, v in sorted(results.items(), key=lambda item: item[0])]
    demos_or_presteps = replace_slot(str(demonstration["response_results"]).replace("'", '"'), {
        "userinput": userinput.replace(conf.TaskTrigger, ""),
        "processes": results
    })
    messages = json.loads(demos_or_presteps)
    messages.insert(0, {"role": "system", "content": hprompt["response_results"]})
    messages.append({"role": "user", "content": fprompt["response_results"]})
    outputstr = chatGPTrequest(messages, False)
    return outputstr


# 任务对模型的映射已经写死在MODELS_MAP里面了， 使用下载数量最高的模型就好了
def chooes_best_model(task):
    choose_reason = "this model is suitable for the task, and it is the most popular one."
    global MODELS_MAP
    mostdownloads, model_id = 0, "init"
    if task == 'text-to-text-generation':
        task = 'text2text-generation'
    for m in MODELS_MAP[task]:
        if mostdownloads < m["downloads"]:
            mostdownloads = m["downloads"]
            model_id = m["id"]
    return model_id, choose_reason


def replace_slot(text, entries):
    for key, value in entries.items():
        if not isinstance(value, str):
            value = str(value)
        text = text.replace("{{" + key + "}}", value.replace('"', "'").replace('\n', ""))
    return text


# if taskjson["task"] == "object-detection":
def object_detection(taskjson, inference):
    img_data = image_to_bytes(taskjson["args"]["image"])
    predicted = inference(data=img_data)
    image = Image.open(io.BytesIO(img_data))
    draw = ImageDraw.Draw(image)
    labels = list(item['label'] for item in predicted)
    color_map = {}
    for label in labels:
        if label not in color_map:
            color_map[label] = (random.randint(0, 255), random.randint(0, 100), random.randint(0, 255))
    for label in predicted:
        box = label["box"]
        draw.rectangle(((box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])), outline=color_map[label["label"]],
                       width=2)
        draw.text((box["xmin"] + 5, box["ymin"] - 15), label["label"], fill=color_map[label["label"]])
    name = str(time.time()).split(".")[0][3:]
    image.save(f"./outputres/{name}.jpg")
    result = {}
    result["generated image with predicted box"] = f"./outputres/{name}.jpg"
    result["predicted"] = predicted
    return result


# taskjson["task"] == "image-segmentation":
def image_segmentation(taskjson, inference):
    image = Image.open(io.BytesIO(image_to_bytes(taskjson["args"]["image"])))
    predicted = inference(data=image_to_bytes(taskjson["args"]["image"]))
    colors = []
    for i in range(len(predicted)):
        colors.append((random.randint(100, 255), random.randint(100, 255), random.randint(100, 255), 155))
    for i, pred in enumerate(predicted):
        mask = base64.b64decode(pred.pop("mask").encode("utf-8"))
        mask = Image.open(io.BytesIO(mask), mode='r')
        mask = mask.convert('L')

        layer = Image.new('RGBA', mask.size, colors[i])
        image.paste(layer, (0, 0), mask)
    name = str(time.time()).split(".")[0][3:]
    image.save(f"./outputres/{name}.jpg")
    result = {}
    result["generated image with segmentation mask"] = f"./outputres/{name}.jpg"
    result["predicted"] = predicted
    return result


def image_to_image(taskjson, model_id):
    response = requests.post(f"https://api-inference.huggingface.co/models/{model_id}",
                             json={"img_url": taskjson["args"]["image"]})
    results = response.json()
    if "path" in results:
        results["generated image"] = results.pop("path")
    return results


def text_to_image(taskjson, inference):
    inputs = taskjson["args"]["text"]
    img = inference(inputs)
    name = str(time.time()).split(".")[0][3:]
    img.save(f"./outputres/{name}.png")
    result = {}
    result["generated image"] = f"./outputres/{name}.png"
    return result


# if taskjson["task"] == "text-to-speech":
def text_to_speech(taskjson, inference):
    response = inference(taskjson["args"]["text"], raw_response=True)
    name = str(time.time()).split(".")[0][3:]
    with open(f"./outputres/{name}.flac", "wb") as f:
        f.write(response.content)
    result = {"generated audio": f"./outputres/{name}.flac"}
    return result


# if taskjson["task"] == "image-to-text":
def image_to_text(taskjson, model_id):
    img_data = image_to_bytes(taskjson["args"]["image"])
    HUGGINGFACE_HEADERS["Content-Length"] = str(len(img_data))
    r = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=HUGGINGFACE_HEADERS,
                      data=img_data, proxies={"https": conf.Proxy})
    result = {}
    if "generated_text" in r.json()[0]:
        result["generated text"] = r.json()[0].pop("generated_text")
    return result


# elif taskjson["task"] == "visual-question-answering" or taskjson["task"] == "document-question-answering":
def visual_question_answering(taskjson, model_id):
    img_url = taskjson["args"]["image"]
    text = taskjson["args"]["text"]
    img_data = image_to_bytes(img_url)
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    json_data = {}
    json_data["inputs"] = {}
    json_data["inputs"]["question"] = text
    json_data["inputs"]["image"] = img_base64
    r = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", json=json_data,
                      headers=HUGGINGFACE_HEADERS, proxies={"https": conf.Proxy})
    result = {}
    if "generated_text" in r.json()[0]:
        result["generated text"] = r.json()[0].pop("generated_text")
    elif "answer" in r.json()[0]:
        result["generated text"] = r.json()[0].pop("answer")
    return result


def get_token_ids_for_task_parsing():  # 模型写死为text-davinci-003
    text = '''{"task": "text-classification",  "token-classification", "text2text-generation", "summarization", "translation",  "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "visual-question-answering", "document-question-answering", "image-segmentation", "text-to-speech", "text-to-video", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "canny-control", "hed-control", "mlsd-control", "normal-control", "openpose-control", "canny-text-to-image", "depth-text-to-image", "hed-text-to-image", "mlsd-text-to-image", "normal-text-to-image", "openpose-text-to-image", "seg-text-to-image", "args", "text", "path", "dep", "id", "<GENERATED>-"}'''
    return list(set(tiktoken.get_encoding("p50k_base").encode(text)))


def request_json(userinput, isStep1):
    global hprompt
    data = {
        "model": conf.ModelName,
        "temperature": conf.temperature,
        "logit_bias": {item: conf.logit_bias for item in get_token_ids_for_task_parsing()},
    }

    if isStep1:
        global demonstration
        messages = demonstration["parse_task"]
        userinput[0]["content"] = fprompt["parse_task"].replace("{{userinput}}", userinput[0]["content"])
        messages += userinput
        if messages[0]['role'] == "system":
            hprompt = messages[0]['content']
            messages = messages[1:]
        final_prompt = hprompt["parse_task"]
        for message in messages:
            final_prompt += f"<im_start>{message['role']}\n{message['content']}<im_end>\n"
    else:
        final_prompt = """"""
        for message in userinput:
            final_prompt += f"<im_start>{message['role']}\n{message['content']}<im_end>\n"

    final_prompt += "<im_start>assistant"
    data["prompt"] = final_prompt
    data['stop'] = data.get('stop', ["<im_end>"])
    data['max_tokens'] = data.get('max_tokens',
                                  max(4096 - len(tiktoken.get_encoding("p50k_base").encode(final_prompt)), 1))
    return data


def chatGPTrequest(userinput, isStep1=True):
    """
    userinput 这个列表中其实只有一个元素
    :param userinput: [{}]
    :return:
    """
    HEADER = {"Authorization": f"Bearer {conf.OpenAiKey}"}
    data = request_json(userinput, isStep1=isStep1)
    response = requests.post(f"https://api.openai.com/v1/{conf.ApiName}", json=data, headers=HEADER,
                             proxies={"https": conf.Proxy})
    if "error" in response.json():
        return response.json()
    return response.json()["choices"][0]["text"].strip()


def collect_result(taskjson, choose, inference_result):
    result = {"taskjson": taskjson}
    result["inference result"] = inference_result
    result["choose model result"] = choose
    return result


def image_to_bytes(img_url):
    img_byte = io.BytesIO()
    type = img_url.split(".")[-1]
    if type == "jpg":
        type = "jpeg"
    load_image(img_url).save(img_byte, format=type)
    img_data = img_byte.getvalue()
    return img_data


if __name__ == '__main__':
    # subtasks = [{"task": "text-to-image", "id": 1, "dep": [-1], "args": {"text": "a photo of cat"}},
    #             {"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "a photo of cow"}}, ]

    # output = step2(subtasks)

    output = {0: {
        'taskjson': {'task': 'conversational', 'id': 0, 'dep': [-1], 'args': {'text': 'please show me a joke of cat'}},
        'inference result': {
            'generated text': 'Q: What do you call a cat that gets everything it wants?\nA: Purr-fect!'},
        'choose model result': {'id': 'ChatGPT', 'reason': 'ChatGPT performs well on some NLP tasks as well.'}},
        1: {'taskjson': {'task': 'text-to-image', 'id': 1, 'dep': [-1], 'args': {'text': 'a photo of cat'}},
            'inference result': {'generated image': './outputres/1668737.png'},
            'choose model result': {'id': 'runwayml/stable-diffusion-v1-5',
                                    'reason': 'this model is suitable for the task, and it is the most popular one.'}}}
    nn = step4("[TASK]please show me a joke and an image of cat", output)
    print("[ Bot ]: ", nn)

    # print("pass")
