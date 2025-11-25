from openai import OpenAI
import os
import json
from datetime import datetime
import re

# API配置 - 提取为公共变量，从环境变量获取
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

def extract_financial_data(pdf_path, company_name="某公司", year="2024"):
    """
    使用Qwen3-VL-Plus模型从PDF中提取全面的财务数据，包括三大报表及附注
    """
    # 读取PDF文件，可能需要将PDF转为图像格式
    # 由于OpenAI API不直接支持PDF，我们需要将PDF的特定页面转为图像
    import fitz  # PyMuPDF
    import base64
    from io import BytesIO
    from PIL import Image

    # 打开PDF文件
    doc = fitz.open(pdf_path)
    
    # 提取所有页面的财务数据
    financial_images = []
    # 遍历PDF文档的所有页面
    for page_num in range(len(doc)):
        page = doc[page_num]
        # 渲染页面为图像
        mat = fitz.Matrix(2.0, 2.0)  # 2倍分辨率
        pix = page.get_pixmap(matrix=mat)
        # 转换为PIL图像
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        # 保存到字节流
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        # 转换为base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        financial_images.append(img_base64)
    
    doc.close()

    # 构建请求内容
    content_parts = []
    
    # 添加全面财务数据提取指令 - 使用传入的公司名称和年份
    content_parts.append({"type": "text", "text": f"请从以下页面中提取{company_name}{year}年度的完整财务信息，需要识别并提取：\n1. 合并资产负债表（包括资产、负债和股东权益各项目及金额）\n2. 合并利润表（营业收入、营业成本、净利润等）\n3. 合并现金流量表（经营活动、投资活动、筹资活动现金流）\n4. 财务报表附注关键信息（应收账款账龄结构、存货构成、固定资产折旧政策、借款明细、资本承诺、或有负债等）\n\n请以结构化格式返回数据，包含所有主要科目和金额、金额单位。"})
    
    # 添加图像数据
    for i, img_base64 in enumerate(financial_images):
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}"
            }
        })

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复
    enable_thinking = True

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": content_parts,
            },
        ],
        stream=True,
        # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
        extra_body={
            'enable_thinking': enable_thinking,
            "thinking_budget": 81920},
    )

    if enable_thinking:
        print("\n" + "=" * 20 + "模型思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "模型回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)
    
    return answer_content

def perform_cash_flow_analysis(extracted_data, company_name="某公司", year="2024"):
    """
    使用qwen3-max对提取的财务数据进行现金流分析
    """
    print(f"正在使用qwen3-max对{company_name}{year}年报进行现金流分析...")
    
    # 使用公共API配置创建第二个客户端用于现金流分析
    analysis_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    # 构建现金流分析指令
    analysis_prompt = f"""
    基于以下从{company_name}{year}年报中提取的财务数据，请执行详细的现金流测算分析：

    {extracted_data}

    请按照现金流测算方案进行以下分析：

    1. 历史现金流分析（间接法）
       - 从净利润开始，调整非现金项目（如折旧摊销）
       - 调整营运资本变动（应收账款、存货、应付账款等）
       - 计算经营活动现金流(CFO)

    2. 关键现金流指标计算
       - 经营活动现金流(CFO)
       - 自由现金流(FCF) = CFO - 资本性支出
       - 债务偿还覆盖率(DSCR) = (CFO - 必要CAPEX) ÷ (当期应还本金 + 利息)
       - 利息保障倍数 = EBIT ÷ 利息支出
       - 现金比率 = (货币资金 + 短期理财) ÷ 短期债务

    3. 附注信息校准
       - 分析应收账款账龄结构与回收风险
       - 评估固定资产折旧政策与资本承诺
       - 检查借款到期结构与还款计划
       - 识别或有负债与租赁义务

    4. 未来现金流预测（直接法）
       - 预测现金流入（主营业务收入、其他经营收入）
       - 预测现金流出（资产购置、运营成本、债务本息）
       - 输出月度/季度滚动现金流预测表

    5. 风险控制与压力测试
       - 收入下降10%-20%情景下的DSCR测试
       - 关键变量的敏感性分析
       - 交叉验证净利润与经营活动现金流的关系

    请提供详细的现金流测算报告，包含具体数值、分析结论和风险提示。
    """
    
    content_parts = [
        {"type": "text", "text": analysis_prompt}
    ]
    
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复
    enable_thinking = True

    # 创建聊天完成请求 - 使用qwen3-max进行分析
    completion = analysis_client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "user",
                "content": content_parts,
            },
        ],
        stream=True,
        # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
        extra_body={
            'enable_thinking': enable_thinking,
            "thinking_budget": 81920},
    )

    if enable_thinking:
        print("\n" + "=" * 20 + "模型分析思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "现金流分析报告" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

    print("\n" + "=" * 20 + "完整现金流分析报告" + "=" * 20 + "\n")
    print(answer_content)
    
    return answer_content

def generate_cash_flow_report_template(extracted_data, company_name="某公司", year="2024"):
    """
    根据提取的财务数据生成符合模板格式的现金流分析报告
    """
    print(f"\n正在生成{company_name}{year}现金流测算分析报告...")
    
    # 创建一个客户端用于报告生成
    report_client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    
    # 构建报告生成指令
    report_prompt = f"""
    基于以下从{company_name}{year}年报中提取的财务数据，请生成一份符合现金流测算分析报告模板格式的详细报告：

    {extracted_data}

    请按照以下模板结构生成报告：

    # **{company_name}{year}企业现金流测算分析报告**

    ## 一、报告基本信息
    - **报告主体**：{company_name}
    - **报告期间**：{year}年度
    - **编制日期**：{datetime.now().strftime('%Y年%m月%d日')}
    - **报告用途**：内部资金管理、融资决策、投资者沟通、预算控制

    ## 二、核心假设与测算方法
    ### 2.1 测算方法
    - **历史分析**：采用间接法，从净利润出发调整非现金项目及营运资本变动
    - **未来预测**：采用直接法，基于合同/订单/预算逐项预测现金流入与流出
    - **压力测试**：设定悲观/基准/乐观多情景，评估现金流韧性

    ### 2.2 关键假设说明
    根据财务数据，请分析并填写：
    | 项目 | 假设内容 |
    |------|----------|
    | 收入增长率 | 基于前一年到{year}年的变化 |
    | 成本结构 | 营业成本占收入比例 |
    | 资本开支（CAPEX） | 固定资产投资情况 |
    | 应收账款周转天数 | 基于{year}年数据 |
    | 应付账款周转天数 | 基于{year}年数据 |
    | 融资安排 | 借款变动情况 |
    | 税率 | 实际税率 |

    ## 三、历史现金流分析（间接法）
    ### 3.1 经营活动现金流（CFO）调节表（单位：元）
    根据利润表和资产负债表数据，构建调节表：
    | 项目 | 金额 |
    |------|------|
    | 净利润 | [从利润表获取] |
    | + 折旧与摊销 | [从现金流量表或附注获取] |
    | + 资产减值损失 | [从利润表获取] |
    | + 财务费用（利息支出等） | [从利润表获取] |
    | - 投资收益 | [从利润表获取] |
    | + 营运资本变动： | |
    | – 应收账款增加 | [资产负债表期初期末变动] |
    | – 存货增加 | [资产负债表期初期末变动] |
    | + 应付账款增加 | [资产负债表期初期末变动] |
    | + 预收款项增加 | [资产负债表期初期末变动] |
    | **经营活动现金流净额（CFO）** | **[计算得出]** |

    ## 四、关键现金流指标分析
    计算并填写：
    | 指标 | 计算公式 | {year}年值 | 健康阈值 | 评价 |
    |------|----------|--------|----------|------|
    | **经营活动现金流（CFO）** | — | [金额] | >0 | [评价] |
    | **自由现金流（FCF）** | CFO – 资本性支出 | [金额] | >0 | [评价] |
    | **CFO / 净利润** | — | [比率] | ≥0.8 | [评价] |
    | **现金比率** | （货币资金 + 短期理财） / 流动负债 | [比率] | ≥0.5 | [评价] |
    | **现金周转期** | 应收天数 + 存货天数 – 应付天数 | [天数] | 越短越好 | [评价] |
    | **债务覆盖率（DSCR）** | FCF / （当年到期本金 + 利息） | [比率] | ≥1.2 | [评价] |

    ## 五、营运资金与资本支出校准
    ### 5.1 应收账款分析
    - 账龄结构：[从附注获取]
    - 坏账准备率：[计算]
    - 主要客户集中度：[从附注获取]

    ### 5.2 存货构成与周转
    - 原材料 / 在产品 / 产成品占比：[从附注获取]
    - 存货周转天数：[计算]

    ### 5.3 资本性支出（CAPEX）
    - 本期实际支出：[从现金流量表获取]
    - 已签约未执行资本承诺：[从附注获取]
    - 主要投向：[从附注获取]

    ## 六、未来12个月现金流滚动预测（直接法）
    基于历史数据和趋势，构建预测：
    | 月份 | 经营现金流入 | 经营现金流出 | 投资支出 | 融资现金流入 | 融资现金流出 | **期末现金余额** |
    |------|----------------|----------------|----------|----------------|----------------|------------------|
    | {int(year)+1}-01 | [预测] | [预测] | [预测] | [预测] | [预测] | [预测] |
    | ... | ... | ... | ... | ... | ... | ... |
    | **合计/平均** | [合计] | [合计] | [合计] | [合计] | [合计] | [合计] |

    ## 七、风险情景与压力测试
    ### 7.1 情景设定
    | 情景 | 描述 |
    |------|------|
    | **基准情景** | 基于历史趋势预测 |
    | **轻度压力** | 收入下降10%，回款延迟15天 |
    | **重度压力** | 收入下降20%，主要客户违约 |

    ### 7.2 压力测试结果（关键指标）
    | 指标 | 基准 | 轻度压力 | 重度压力 |
    |------|------|----------|----------|
    | CFO（万元） | [计算] | [计算] | [计算] |
    | 期末现金余额（万元） | [计算] | [计算] | [计算] |
    | DSCR | [计算] | [计算] | [计算] |

    ### 7.3 敏感性分析
    - **收入每下降1%** → CFO 减少约 [金额] 万元
    - **应收账款周转天数增加10天** → 现金流减少 [金额] 万元

    ## 八、结论与行动建议
    ### 8.1 主要结论
    - 当前现金流状况总体[健康/紧张]，主要分析现金流状况
    - 现金余额可覆盖未来 X 个月刚性支出
    - 风险点分析

    ### 8.2 管理建议
    1. **加快回款**：具体建议
    2. **优化付款节奏**：具体建议
    3. **控制非必要CAPEX**：具体建议
    4. **建立应急融资渠道**：具体建议
    5. **设定预警机制**：具体建议

    请根据实际财务数据填写所有[ ]中的具体数值和分析内容。
    """
    
    content_parts = [
        {"type": "text", "text": report_prompt}
    ]
    
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复
    enable_thinking = True

    # 创建聊天完成请求 - 使用qwen3-max生成报告
    completion = report_client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {
                "role": "user",
                "content": content_parts,
            },
        ],
        stream=True,
        # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
        extra_body={
            'enable_thinking': enable_thinking,
            "thinking_budget": 81920},
    )

    if enable_thinking:
        print("\n" + "=" * 20 + "报告生成思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "现金流分析报告" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

    print("\n" + "=" * 20 + "完整现金流分析报告" + "=" * 20 + "\n")
    print(answer_content)
    
    return answer_content


def save_report_to_file(report_content, company_name="某公司", year="2024"):
    """
    将生成的现金流分析报告保存到文件
    """
    # 创建报告文件名
    filename = f"{company_name}_{year}_现金流分析报告.md"
    filepath = f"/data/financial_analysis/{filename}"
    
    # 保存报告内容到文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n现金流分析报告已保存至: {filepath}")
    return filepath


def cash_flow_analysis(pdf_path, company_name="某公司", year="2024"):
    """
    完整的现金流分析流程：提取数据 -> 现金流测算分析 -> 生成报告
    """
    print(f"开始对{company_name}{year}年报进行现金流测算...")
    
    # 步骤1: 使用qwen3-vl-plus提取财务数据
    print(f"\n步骤1: 正在使用qwen3-vl-plus提取{company_name}{year}年报的财务数据...")
    extracted_data = extract_financial_data(pdf_path, company_name, year)
    
    # 步骤2: 使用qwen3-max进行现金流分析
    print(f"\n步骤2: 正在使用qwen3-max对{company_name}{year}年报进行现金流分析...")
    analysis_result = perform_cash_flow_analysis(extracted_data, company_name, year)
    
    # 步骤3: 生成符合模板格式的现金流分析报告
    print(f"\n步骤3: 正在生成{company_name}{year}现金流测算分析报告...")
    report_content = generate_cash_flow_report_template(extracted_data, company_name, year)
    
    # 步骤4: 保存报告到文件
    print(f"\n步骤4: 保存现金流分析报告...")
    report_filepath = save_report_to_file(report_content, company_name, year)
    
    # 步骤5: 输出最终结果
    print(f"\n=== 现金流测算完成 ===")
    print(f"已成功完成对{company_name}{year}年报的现金流测算分析")
    print(f"分析报告已保存至: {report_filepath}")
    
    return {
        'analysis_result': analysis_result,
        'report_content': report_content,
        'report_filepath': report_filepath
    }

if __name__ == "__main__":
    # 默认分析交银金租2024年报，可以修改为其他公司
    #pdf_path = "/data/financial_analysis/中旅国际2024年报（数据加附注）.pdf"
    #company_name = "中旅国际"
    pdf_path = "/data/financial_analysis/交银金租2024年报（数据加附注）.pdf"
    company_name = "交银金租"
    year = "2024"
    
    cash_flow_analysis(pdf_path, company_name, year)

