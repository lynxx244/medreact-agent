# react_agent.py
import re
from openai import OpenAI  # DeepSeek 兼容 OpenAI 接口

client = OpenAI(
    api_key="your_deepseek_api_key_here",
    base_url="https://api.deepseek.com"
)

# =========================================
# 第一部分：定义工具
# =========================================

# 全局变量，控制是否为评估模式
EVAL_MODE = False

def ask_patient(question: str) -> str:
    if EVAL_MODE:
        return "没有其他特殊症状。"  # 评估时自动回答
    print(f"\n[Agent 追问]: {question}")
    return input("[患者回答]: ")
def risk_assess(symptoms: list, duration_days: int) -> str:
    """用LLM判断是否有红旗症状，再结合持续时间给出风险等级"""

    # 第一步：让LLM判断是否有危险症状
    symptom_str = "、".join(symptoms)

    check_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"""你是一个急诊分诊专家。
判断以下症状是否属于需要立即（2小时内）急诊处理的红旗症状。

红旗症状的标准（必须满足其中之一）：
- 生命体征异常：意识丧失、休克、呼吸困难、大量出血
- 急性心脑血管：胸痛伴大汗、突发剧烈头痛、口眼歪斜、半身不遂
- 急腹症：剧烈腹痛伴板状腹、腹部外伤
- 严重过敏：全身荨麻疹伴喉咙水肿

以下不属于红旗症状：
- 慢性病就诊、复查需求
- 普通感染（股癣、普通腹泻等）
- 月经异常、普通妇科问题
- 术后常规随访（无发热、无大量出血）

症状列表：{symptom_str}

只回答以下格式：
危险症状：有
原因：[具体符合哪条红旗标准]

或者：
危险症状：无
原因：无"""
        }],
        temperature=0.0  # 判断题用0温度，要最确定的答案
    )

    result = check_response.choices[0].message.content
    print(f"[risk_assess LLM返回]: {result}")

    # 第二步：根据LLM判断结果决定风险等级
    if "危险症状：有" in result or "危险症状：[有]" in result:
        # 提取原因
        reason = result.split("原因：")[-1].strip()
        return f"风险等级：高，{reason}，建议立即就医或拨打120"

    # 没有红旗症状，再看持续时间
    if duration_days > 7:
        return "风险等级：中等，症状持续较长，建议尽快就医"
    elif duration_days > 3:
        return "风险等级：中等偏低，建议就近就医观察"
    else:
        return "风险等级：较低，可先在家观察，症状加重立即就医"
def search_symptom(query: str) -> str:
    """查询症状相关信息（第一周让LLM模拟返回）"""
    # 后期可以接真实医学知识库
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"作为医学知识库，简洁回答：{query}，只给事实，不给建议，50字以内"
        }]
    )
    return response.choices[0].message.content

# 工具注册表：名字 → 函数
TOOLS = {
    "ask_patient": ask_patient,
    "risk_assess": risk_assess,
    "search_symptom": search_symptom,
}

TOOLS_DESCRIPTION = """
你可以使用以下工具：
- ask_patient(question): 向患者追问症状细节，question 是你要问的问题
- risk_assess(symptoms, duration_days): 评估风险，symptoms 是症状列表，duration_days 是持续天数
- search_symptom(query): 查询症状相关医学信息

调用格式必须严格如下：
Action: 工具名(参数)
例如：
Action: ask_patient("你有发烧吗？")
Action: risk_assess(["发烧", "咳嗽"], 3)
Action: search_symptom("儿童发烧超过38.5度的常见原因")
"""

# =========================================
# 第二部分：Prompt 构建
# =========================================

SYSTEM_PROMPT = f"""你是一个医疗预问诊 Agent。
你的任务是通过多轮推理和工具调用，收集患者症状信息，给出初步分诊建议。

{TOOLS_DESCRIPTION}

你必须按照以下格式输出，每次只输出一个 Thought + 一个 Action，或者给出 Final Answer：

格式一（需要继续收集信息）：
Thought: [你的推理过程]
Action: [工具调用]

格式二（信息足够，给出答案）：
Thought: [你的最终推理]
Final Answer: [给患者的建议]

注意：
-每次 Action 只能调用 ask_patient 一次，且问题只包含一个句号。
正确示例：Action: ask_patient("头痛持续几天了？")
错误示例：Action: ask_patient("持续几天了？有没有发烧？")
- 风险评估前必须先收集足够信息
-在最终回答之前至少调用一次risk_assess，防止有遗漏的紧急情况没发现
- Final Answer 必须严格按照以下格式输出，不能省略第一行：
风险等级：高/中/低
可能原因：xxx
建议行动：xxx
注意事项：xxx
"""

def build_prompt(question: str, history: list) -> list:
    """把对话历史拼成发给LLM的messages"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": f"患者描述：{question}"})
    # 加入历史轮次
    for item in history:
        messages.append({"role": "assistant", "content": item["agent"]})
        messages.append({"role": "user", "content": f"Observation: {item['observation']}"})

    # 加入当前问题（只在第一轮加）
    if not history:
        messages.append({"role": "user", "content": f"患者描述：{question}"})

    return messages

# =========================================
# 第三部分：输出解析
# =========================================

def parse_output(text: str):
    """
    解析 LLM 输出，返回 (thought, action_name, action_input, is_final, final_answer)
    这是最容易出 bug 的地方，要写得健壮
    """
    # 检查是否是最终答案
    if "Final Answer:" in text:
        thought = ""
        if "Thought:" in text:
            thought = text.split("Thought:")[1].split("Final Answer:")[0].strip()
        final_answer = text.split("Final Answer:")[1].strip()
        return thought, None, None, True, final_answer

    # 解析 Action
    if "Action:" in text:
        thought = ""
        if "Thought:" in text:
            thought = text.split("Thought:")[1].split("Action:")[0].strip()

        action_str = text.split("Action:")[1].strip()

        # 解析工具名和参数，例如：ask_patient("你发烧吗？")
        match = re.match(r'(\w+)\((.*)\)', action_str, re.DOTALL)
        if match:
            action_name = match.group(1)
            action_input_str = match.group(2).strip()
            return thought, action_name, action_input_str, False, None

    # 解析失败（LLM输出格式乱了）
    return text, None, None, False, None

def execute_tool(action_name: str, action_input_str: str) -> str:
    """执行工具调用"""
    if action_name not in TOOLS:
        return f"错误：工具 '{action_name}' 不存在"

    try:
        # 用 eval 解析参数（简单粗暴，第一周够用，后期要改）
        result = eval(f"TOOLS['{action_name}']({action_input_str})")
        return str(result)
    except Exception as e:
        return f"工具执行失败：{e}"

# =========================================
# 第四部分：主循环
# =========================================

class MedReActAgent:
    def __init__(self, max_steps=8):
        self.max_steps = max_steps

    def run(self, patient_input: str):
        history = []
        print(f"\n{'='*50}")
        print(f"患者：{patient_input}")
        print(f"{'='*50}")

        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")

            # 1. 构建 prompt
            messages = build_prompt(patient_input, history)

            # 2. 调用 LLM
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1  # 低温度，让输出更稳定
            )
            output = response.choices[0].message.content
            print(f"LLM 输出：\n{output}")

            # 3. 解析输出
            thought, action_name, action_input_str, is_final, final_answer = parse_output(output)

            # 4. 如果是最终答案，结束
            if is_final:
                # 安全兜底：如果准备给出低/中风险结论，强制二次校验
                if "风险等级：低" in final_answer or "风险等级：中" in final_answer:
                    safety_check = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{
                            "role": "user",
                            "content": f"""你是急诊安全专家，专门负责最后一道安全审查。
            患者原始描述：{patient_input}
            Agent初步结论：{final_answer[:200]}

            请判断是否存在以下任何一种被忽视的紧急信号：
            - 出血相关：便血、咯血、呕血、大量出血
            - 神经相关：意识模糊、突发剧烈头痛、口眼歪斜、肢体无力
            - 心肺相关：胸痛、呼吸困难、心跳异常
            - 术后并发症：术后发热、伤口感染、持续出血
            - 肿瘤相关：已知肿瘤伴新发症状
            注意：以下情况不应升级为高风险：
            - 慢性病、皮肤病（股癣、湿疹等）久治不愈
            - 月经异常但无大量出血
            - 普通消化道症状（轻度腹泻、便秘）
            - 症状轻微且持续时间短
            只有明确符合上述紧急信号之一才回答"需要升级：是"

            只回答：
            需要升级：是
            原因：[具体原因]

            或者：
            需要升级：否"""
                        }],
                        temperature=0.0,
                    )
                    safety_result = safety_check.choices[0].message.content
                    print(f"[安全校验]: {safety_result[:80]}")

                    if "需要升级：是" in safety_result:
                        reason = safety_result.split("原因：")[-1].strip()
                        final_answer = f"风险等级：高\n可能原因：{reason}\n建议行动：建议立即前往医院急诊科就诊或拨打120。\n注意事项：请勿延误，尽快就医。"

                print(f"\n{'='*50}")
                print(f"[分诊结果]\n{final_answer}")
                print(f"{'='*50}")
                return final_answer

            # 5. 如果有工具调用，执行并记录
            if action_name:
                observation = execute_tool(action_name, action_input_str)
                print(f"Observation: {observation}")
                history.append({
                    "agent": output,
                    "observation": observation
                })
            else:
                # 输出格式异常，强制结束
                print("警告：LLM输出格式异常，终止")
                return "抱歉，处理过程中出现错误，请重新描述症状。"

        # 步数用完，强制生成最终答案
        print("\n[步数用完，强制生成最终答案]")
        messages = build_prompt(patient_input, history)
        messages.append({
            "role": "user",
            "content": "请根据目前收集到的信息，立即给出Final Answer，不要再调用工具。"
        })
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1
        )
        output = response.choices[0].message.content
        _, _, _, _, final_answer = parse_output(output)
        if final_answer:
            return final_answer
        return "已达到最大推理步数，建议直接就医。"


# =========================================
# 运行入口
# =========================================

if __name__ == "__main__":
    agent = MedReActAgent(max_steps=8)
    patient_input = input("请描述你的症状：")
    agent.run(patient_input)