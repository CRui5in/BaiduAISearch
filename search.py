import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import traceback
import requests
from bs4 import BeautifulSoup
import backoff
import openai
from rag import RAGSystem

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 config.txt 文件的完整路径
config_path = os.path.join(current_dir, 'config.txt')

# 读取配置文件
config = {}
with open(config_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

rag_system = RAGSystem(
    api_key=config['OPENAI_API_KEY'],
    base_url=config['OPENAI_BASE_URL']
)
rag_system.load_index()

# 设置环境变量和配置
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_BASE_URL'] = config['OPENAI_BASE_URL']

# 打印环境变量进行调试
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")
print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL')}")

# -----------------------------------------------------------------------------
# 默认配置和提示
NUM_SEARCH = int(config.get('NUM_SEARCH', '20'))
SEARCH_TIME_LIMIT = int(config.get('SEARCH_TIME_LIMIT', '3'))
TOTAL_TIMEOUT = int(config.get('TOTAL_TIMEOUT', '6'))
MAX_CONTENT = int(config.get('MAX_CONTENT', '1000'))
MAX_TOKENS = int(config.get('MAX_TOKENS', '2000'))
LLM_MODEL = config.get('LLM_MODEL', 'wno4y6gl_adsad')

system_prompt_search = """你是一个乐于助人的AI助手，其主要目标是决定用户的查询是否需要百度搜索。"""
search_prompt = """
决定用户的查询是否需要百度搜索。您应该使用百度搜索大多数查询，以找到最准确和最新的信息。遵循以下条件：

-如果查询不需要百度搜索，则必须输“不需要百度搜索”。
-如果查询需要百度搜索，您必须以重新制定的百度搜索用户查询作为回应。
-用户查询有时可能会引用以前的消息。确保你的百度搜索考虑了整个消息历史。

用户查询：
{query}
"""

system_prompt_answer = """你是一位乐于助人的AI助手，擅长回答用户的问题"""
answer_prompt = """生成一个信息丰富且与用户查询相关的响应
用户查询：
{query}
"""

system_prompt_cited_answer = """你是一个乐于助人的助手，擅长根据引用的上下文回答用户的问题。"""
cited_answer_prompt = """
使用给定的上下文（带有[引文编号]（网站链接）和简要描述的搜索结果）对用户的查询提供相关、信息丰富的回复。
-直接回答，无需向用户推荐任何外部链接。
-使用公正的新闻语气，避免重复文本。
-为了清楚起见，请用带有要点的标记来格式化您的回复。
-使用[引用号]（网站链接）符号引用所有信息，将答案的每一部分与其来源相匹配。
上下文块：
{context_block}

用户查询：
{query}
"""
# -----------------------------------------------------------------------------

# 设置OpenAI API
client = openai.OpenAI(
    api_key=config['OPENAI_API_KEY'],
    base_url=config['OPENAI_BASE_URL']
)


def trace_function_factory(start):
    """创建超时请求的跟踪函数"""

    def trace_function(frame, event, arg):
        if time.time() - start > TOTAL_TIMEOUT:
            raise TimeoutError('网站获取超时')
        return trace_function

    return trace_function


def fetch_webpage(url, timeout):
    """获取给定URL和超时的网页内容。"""
    start = time.time()
    sys.settrace(trace_function_factory(start))
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        response.encoding = 'utf-8'  # 确保使用UTF-8编码
        soup = BeautifulSoup(response.text, 'lxml')
        title = soup.title.string if soup.title else "无标题"
        paragraphs = soup.find_all('p')
        page_text = ' '.join([para.get_text() for para in paragraphs])
        return url, title, page_text
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"获取 {url} 时出错: {e}", file=sys.stderr)
    finally:
        sys.settrace(None)
    return url, None, None


def baidu_search(query, num_results=10):
    """执行百度搜索并返回结果"""
    print(f"执行百度搜索: {query}")  # 添加调试信息
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = f"https://www.baidu.com/s?wd={query}&rn={num_results}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        print(f"百度搜索状态码: {response.status_code}")  # 添加状态码调试信息
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = []
        for result in soup.select('.result.c-container'):
            link = result.select_one('h3.t a')
            if link and 'href' in link.attrs:
                search_results.append(link['href'])
        print(f"百度搜索结果数量: {len(search_results)}")  # 添加调试信息
        if len(search_results) == 0:
            print("警告：没有找到搜索结果。HTML内容：")
            print(response.text[:1000])  # 打印前1000个字符的HTML内容
        return search_results[:num_results]
    except requests.RequestException as e:
        print(f"百度搜索请求失败: {e}")
        return []


def parse_baidu_results(query, num_search=NUM_SEARCH, search_time_limit=SEARCH_TIME_LIMIT):
    """执行百度搜索并解析顶部结果的内容。"""
    print(f"开始搜索: {query}")  # 添加调试信息
    urls = baidu_search(query, num_search)
    print(f"搜索到的URL数量: {len(urls)}")  # 添加调试信息
    max_workers = os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_webpage, url, search_time_limit): url for url in urls}
        results = {url: (title, page_text) for future in as_completed(future_to_url)
                   if
                   (url := future.result()[0]) and (title := future.result()[1]) and (page_text := future.result()[2])}
    print(f"成功获取的网页数量: {len(results)}")  # 添加调试信息
    return results


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def llm_check_search(query, msg_history=None, llm_model=LLM_MODEL):
    """执行百度搜索。"""
    prompt = search_prompt.format(query=query)
    msg_history = msg_history or []
    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": system_prompt_search}, *new_msg_history],
        max_tokens=30
    ).choices[0].message.content

    # 返回 LLM 的响应
    return response


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def llm_answer(query, msg_history=None, search_dic=None, llm_model=LLM_MODEL, max_content=MAX_CONTENT,
               max_tokens=MAX_TOKENS):
    """生成包含搜索结果上下文的语言模型提示。"""
    if search_dic:
        context_block = "\n".join([f"[{i + 1}]({url}) {title}: {content[:max_content]}"
                                   for i, (url, (title, content)) in enumerate(search_dic.items())])
        prompt = cited_answer_prompt.format(context_block=context_block, query=query)
        system_prompt = system_prompt_cited_answer
    else:
        prompt = answer_prompt.format(query=query)
        system_prompt = system_prompt_answer

    """使用带有流完成的OpenAI语言模型生成响应"""
    msg_history = msg_history or []
    new_msg_history = msg_history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": system_prompt}, *new_msg_history],
        max_tokens=max_tokens,
        stream=True
    )

    content = []
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            content.append(chunk_content)

    new_msg_history = new_msg_history + [{"role": "assistant", "content": ''.join(content)}]
    return new_msg_history

def process_query(query):
    """处理查询并返回结果"""
    try:
        msg_history = None
        search_result = llm_check_search(query, msg_history)

        # 获取百度搜索结果
        search_dic = parse_baidu_results(search_result)

        # 获取RAG检索结果
        rag_results = rag_system.search(query, top_k=3)

        # 合并RAG结果到搜索字典
        for i, result in enumerate(rag_results):
            doc_id = f"rag_{i}"
            search_dic[doc_id] = (
                f"RAG Document {i + 1}",
                result['content']
            )

        msg_history = llm_answer(query, msg_history, search_dic)

        response = {
            "answer": msg_history[-1]["content"] if msg_history else "未找到结果",
            "sources": []
        }

        # 添加所有来源
        for i, (url, (title, _)) in enumerate(search_dic.items(), start=1):
            response["sources"].append({
                "index": i,
                "title": title,
                "url": url if not url.startswith("rag_") else "Local RAG Document"
            })

        return response
    except Exception as e:
        error_response = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return error_response


def main(query):
    """主要功能是执行搜索、生成响应并返回结果。"""
    try:
        response = process_query(query)
        # 只输出JSON结果
        print(json.dumps(response, ensure_ascii=False))
    except Exception as e:
        error_response = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        # 将错误信息输出到标准错误
        print(json.dumps(error_response, ensure_ascii=False), file=sys.stderr)
        # 确保标准输出有一个有效的JSON
        print(json.dumps({"error": "发生内部错误"}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        main(query)
    else:
        print(json.dumps({"error": "请提供搜索查询作为命令行参数"}, ensure_ascii=False))
