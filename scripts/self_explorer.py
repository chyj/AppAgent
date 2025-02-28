import argparse
import ast
import datetime
import json
import os
import re
import sys
import time

import prompts
from config import load_config
from and_controller import list_all_devices, AndroidController, traverse_tree
from model import parse_explore_rsp, parse_reflect_rsp, OpenAIModel, QwenModel
from utils import print_with_color, draw_bbox_multi

arg_desc = "AppAgent - Autonomous Exploration"
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)
parser.add_argument("--app")
parser.add_argument("--root_dir", default="./")
args = vars(parser.parse_args())

configs = load_config()

if configs["MODEL"] == "OpenAI":
    mllm = OpenAIModel(base_url=configs["OPENAI_API_BASE"],
                       api_key=configs["OPENAI_API_KEY"],
                       model=configs["OPENAI_API_MODEL"],
                       temperature=configs["TEMPERATURE"],
                       max_tokens=configs["MAX_TOKENS"])
elif configs["MODEL"] == "Qwen":
    mllm = QwenModel(api_key=configs["DASHSCOPE_API_KEY"],
                     model=configs["QWEN_MODEL"])
else:
    print_with_color(f"ERROR: Unsupported model type {configs['MODEL']}!", "red")
    sys.exit()

app = args["app"]
root_dir = args["root_dir"]

if not app:
    print_with_color("What is the name of the target app?", "blue")
    app = input()
    app = app.replace(" ", "")

# 创建工作目录
work_dir = os.path.join(root_dir, "apps")
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
work_dir = os.path.join(work_dir, app)
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
demo_dir = os.path.join(work_dir, "demos")
if not os.path.exists(demo_dir):
    os.mkdir(demo_dir)
demo_timestamp = int(time.time())
task_name = datetime.datetime.fromtimestamp(demo_timestamp).strftime("self_explore_%Y-%m-%d_%H-%M-%S")
task_dir = os.path.join(demo_dir, task_name)
os.mkdir(task_dir)
docs_dir = os.path.join(work_dir, "auto_docs")
if not os.path.exists(docs_dir):
    os.mkdir(docs_dir)
explore_log_path = os.path.join(task_dir, f"log_explore_{task_name}.txt")
reflect_log_path = os.path.join(task_dir, f"log_reflect_{task_name}.txt")

# 选择设备
device_list = list_all_devices()
if not device_list:
    print_with_color("ERROR: No device found!", "red")
    sys.exit()
print_with_color(f"List of devices attached:\n{str(device_list)}", "yellow")
if len(device_list) == 1:
    device = device_list[0]
    print_with_color(f"Device selected: {device}", "yellow")
else:
    print_with_color("Please choose the Android device to start demo by entering its ID:", "blue")
    device = input()
controller = AndroidController(device)
width, height = controller.get_device_size()
if not width and not height:
    print_with_color("ERROR: Invalid device size!", "red")
    sys.exit()
print_with_color(f"Screen resolution of {device}: {width}x{height}", "yellow")

print_with_color("Please enter the description of the task you want me to complete in a few sentences:", "blue")
task_desc = input()

# 开始探索
# 初始化变量
round_count = 0
doc_count = 0
useless_list = set()
last_act = "None"
task_complete = False
# 探索循环，限制最大轮数
while round_count < configs["MAX_ROUNDS"]:
    round_count += 1
    print_with_color(f"Round {round_count}", "yellow")
    # 获取当前屏幕截图和XML文件
    screenshot_before = controller.get_screenshot(f"{round_count}_before", task_dir)
    xml_path = controller.get_xml(f"{round_count}", task_dir)
    if screenshot_before == "ERROR" or xml_path == "ERROR":
        break
    # 解析XML文件，获取可点击和可聚焦的元素
    clickable_list = []
    focusable_list = []
    traverse_tree(xml_path, clickable_list, "clickable", True)
    traverse_tree(xml_path, focusable_list, "focusable", True)
    elem_list = []
    for elem in clickable_list:
        if elem.uid in useless_list:
            continue
        elem_list.append(elem)
    for elem in focusable_list:
        if elem.uid in useless_list:
            continue
        bbox = elem.bbox
        center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
        close = False
        for e in clickable_list:
            bbox = e.bbox
            center_ = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
            dist = (abs(center[0] - center_[0]) ** 2 + abs(center[1] - center_[1]) ** 2) ** 0.5
            if dist <= configs["MIN_DIST"]:
                close = True
                break
        if not close:
            elem_list.append(elem)
    # 绘制元素框图
    draw_bbox_multi(screenshot_before, os.path.join(task_dir, f"{round_count}_before_labeled.png"), elem_list,
                    dark_mode=configs["DARK_MODE"])
    # 生成探索提示
    prompt = re.sub(r"<task_description>", task_desc, prompts.self_explore_task_template)
    prompt = re.sub(r"<last_act>", last_act, prompt)
    base64_img_before = os.path.join(task_dir, f"{round_count}_before_labeled.png")
    print_with_color("Thinking about what to do in the next step...", "yellow")
    # 获取模型响应
    status, rsp = mllm.get_model_response(prompt, [base64_img_before])

    if status:
        # 记录探索日志
        with open(explore_log_path, "a") as logfile:
            log_item = {"step": round_count, "prompt": prompt, "image": f"{round_count}_before_labeled.png",
                        "response": rsp}
            logfile.write(json.dumps(log_item) + "\n")
        res = parse_explore_rsp(rsp)  
        # 解析模型响应
        act_name = res[0]
        last_act = res[-1]
        res = res[:-1]
        # 根据模型响应执行操作
        if act_name == "FINISH":
            task_complete = True
            break
        # 执行点击操作
        if act_name == "tap":
            _, area = res
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.tap(x, y)
            if ret == "ERROR":
                print_with_color("ERROR: tap execution failed", "red")
                break
        # 执行文本输入操作
        elif act_name == "text":
            _, input_str = res
            ret = controller.text(input_str)
            if ret == "ERROR":
                print_with_color("ERROR: text execution failed", "red")
                break
        # 执行长按操作
        elif act_name == "long_press":
            _, area = res
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.long_press(x, y)
            if ret == "ERROR":
                print_with_color("ERROR: long press execution failed", "red")
                break
        # 执行滑动操作
        elif act_name == "swipe":
            _, area, swipe_dir, dist = res
            tl, br = elem_list[area - 1].bbox
            x, y = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            ret = controller.swipe(x, y, swipe_dir, dist)
            if ret == "ERROR":
                print_with_color("ERROR: swipe execution failed", "red")
                break
        else:
            break
        # 等待请求间隔
        time.sleep(configs["REQUEST_INTERVAL"])
    else:
        print_with_color(rsp, "red")
        break

    # 反射阶段
    # 获取当前屏幕截图
    screenshot_after = controller.get_screenshot(f"{round_count}_after", task_dir)
    if screenshot_after == "ERROR":
        break
    # 绘制元素框图
    draw_bbox_multi(screenshot_after, os.path.join(task_dir, f"{round_count}_after_labeled.png"), elem_list,
                    dark_mode=configs["DARK_MODE"])
    base64_img_after = os.path.join(task_dir, f"{round_count}_after_labeled.png")

    # 生成反射提示
    if act_name == "tap":
        prompt = re.sub(r"<action>", "tapping", prompts.self_explore_reflect_template)
    elif act_name == "text":
        continue
    elif act_name == "long_press":
        prompt = re.sub(r"<action>", "long pressing", prompts.self_explore_reflect_template)
    elif act_name == "swipe":
        swipe_dir = res[2]
        if swipe_dir == "up" or swipe_dir == "down":
            act_name = "v_swipe"
        elif swipe_dir == "left" or swipe_dir == "right":
            act_name = "h_swipe"
        prompt = re.sub(r"<action>", "swiping", prompts.self_explore_reflect_template)
    else:
        print_with_color("ERROR: Undefined act!", "red")
        break
    prompt = re.sub(r"<ui_element>", str(area), prompt)
    prompt = re.sub(r"<task_desc>", task_desc, prompt)
    prompt = re.sub(r"<last_act>", last_act, prompt)

    # 获取模型响应
    print_with_color("Reflecting on my previous action...", "yellow")
    status, rsp = mllm.get_model_response(prompt, [base64_img_before, base64_img_after])
    if status:
        resource_id = elem_list[int(area) - 1].uid
        with open(reflect_log_path, "a") as logfile:
            log_item = {"step": round_count, "prompt": prompt, "image_before": f"{round_count}_before_labeled.png",
                        "image_after": f"{round_count}_after.png", "response": rsp}
            logfile.write(json.dumps(log_item) + "\n")
        res = parse_reflect_rsp(rsp)
        decision = res[0]
        if decision == "ERROR":
            break
        if decision == "INEFFECTIVE":
            useless_list.add(resource_id)
            last_act = "None"
        elif decision == "BACK" or decision == "CONTINUE" or decision == "SUCCESS":
            if decision == "BACK" or decision == "CONTINUE":
                useless_list.add(resource_id)
                last_act = "None"
                if decision == "BACK":
                    ret = controller.back()
                    if ret == "ERROR":
                        print_with_color("ERROR: back execution failed", "red")
                        break
            doc = res[-1]
            doc_name = resource_id + ".txt"
            doc_path = os.path.join(docs_dir, doc_name)
            if os.path.exists(doc_path):
                doc_content = ast.literal_eval(open(doc_path).read())
                if doc_content[act_name]:
                    print_with_color(f"Documentation for the element {resource_id} already exists.", "yellow")
                    continue
            else:
                doc_content = {
                    "tap": "",
                    "text": "",
                    "v_swipe": "",
                    "h_swipe": "",
                    "long_press": ""
                }
            doc_content[act_name] = doc
            with open(doc_path, "w") as outfile:
                outfile.write(str(doc_content))
            doc_count += 1
            print_with_color(f"Documentation generated and saved to {doc_path}", "yellow")
        else:
            print_with_color(f"ERROR: Undefined decision! {decision}", "red")
            break
    else:
        print_with_color(rsp["error"]["message"], "red")
        break
    time.sleep(configs["REQUEST_INTERVAL"])

if task_complete:
    print_with_color(f"Autonomous exploration completed successfully. {doc_count} docs generated.", "yellow")
elif round_count == configs["MAX_ROUNDS"]:
    print_with_color(f"Autonomous exploration finished due to reaching max rounds. {doc_count} docs generated.",
                     "yellow")
else:
    print_with_color(f"Autonomous exploration finished unexpectedly. {doc_count} docs generated.", "red")

    # 1. 这段代码的目的是通过与设备的交互来完成特定任务，并在每次交互后根据响应做出决策。
    # 2. 决策可以分为“无效”、“返回”、“继续”或“成功”。
    # 3. 如果决策是“无效”，则将当前资源ID添加到无效列表，并将最后的动作为"None"。
    # 4. 对于“返回”或“继续”的决策，也将资源ID添加到无效列表，并将最后的动作为"None"。
    #    - 如果决策是“返回”，则调用controller的back()方法尝试返回上一步。
    #    - 如果back()方法失败，则打印错误信息并终止循环。
    # 5. 如果决策是“成功”或“继续”，则生成文档并保存到指定路径。
    #    - 从响应中提取文档内容，并生成文档名称和路径。
    #    - 如果文档已存在，则读取其内容。
    #    - 如果文档中对应动作的条目已存在，输出提示信息并跳过生成步骤。
    #    - 如果文档不存在，初始化一个新的文档内容字典，包含所有可能的动作条目。
    #    - 更新文档内容字典中对应动作的条目，并将其写入文件。
    #    - 增加生成文档的计数。
    # 6. 在每次循环结束时，等待一段时间以符合请求间隔配置。
    # 7. 如果任务完成，输出成功信息，显示生成的文档数量。
    # 8. 如果达到最大轮数，输出相应信息，显示生成的文档数量。
    # 9. 如果在其他情况下结束，输出意外结束信息，显示生成的文档数量。
    # 10. 在这个过程中，LLM用于生成文档内容，基于与设备的交互结果和任务描述。
    # 11. LLM的原理是通过预训练模型生成自然语言文本，帮助自动化文档生成。
