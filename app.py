import os
import gradio as gr
from search import process_query
from rag import RAGSystem
import tempfile
from pathlib import Path
import shutil

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造配置文件路径
config_path = os.path.join(current_dir, 'config.txt')
documents_dir = os.path.join(current_dir, 'documents')

# 确保documents目录存在
os.makedirs(documents_dir, exist_ok=True)

# 读取配置文件
config = {}
with open(config_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()

# 设置环境变量
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_BASE_URL'] = config['OPENAI_BASE_URL']

# 初始化RAG系统
rag_system = RAGSystem(
    api_key=config['OPENAI_API_KEY'],
    base_url=config['OPENAI_BASE_URL']
)
rag_system.load_index()


def format_response(response):
    if "error" in response:
        return f"错误：{response['error']}\n\n堆栈跟踪：\n{response['traceback']}"

    answer = response["answer"]
    sources = response["sources"]

    formatted_sources = "\n".join([f"{s['index']}. [{s['title']}]({s['url']})" for s in sources])

    return f"{answer}\n\n参考来源：\n{formatted_sources}"


def search_and_answer(query):
    result = process_query(query)
    return format_response(result)


def process_uploaded_files(files):
    """处理上传的文件并生成索引"""
    try:
        # 清空documents目录
        for file_path in Path(documents_dir).glob("*"):
            if file_path.is_file():
                file_path.unlink()

        # 保存新上传的文件
        documents = {}
        for file in files:
            # 获取原始文件名
            original_filename = Path(file.name).name
            # 构造目标路径
            target_path = os.path.join(documents_dir, original_filename)

            # 如果是文本文件，直接读取内容
            print(f"开始处理{original_filename}")
            if file.name.endswith('.txt'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents[original_filename] = content

                # 复制文件到documents目录
                shutil.copy2(file.name, target_path)

        # 生成索引
        if documents:
            rag_system.add_documents(documents)
            print(f"处理完毕")
            return f"成功为{len(documents)}个文件生成索引"
        else:
            return "没有找到可处理的文本文件"

    except Exception as e:
        print(f"处理失败")
        return f"处理文件时出错: {str(e)}"


# 修改自定义 CSS
custom_css = """
body {
    background-color: #f0f8ff;
}
.container {max-width: 730px; margin: auto; padding-top: 0;}
.title-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.main-title {
    font-size: 2.5em;
    font-weight: bold;
    color: #1e90ff;
    display: flex;
    align-items: center;
    justify-content: center;
}
.main-title img {
    width: 40px;
    height: 40px;
    margin-right: 10px;
    filter: invert(39%) sepia(95%) saturate(1095%) hue-rotate(194deg) brightness(103%) contrast(101%);
}
.powered-by {
    font-size: 0.8em;
    color: #4682b4;
    margin-top: 0.5rem;
}
.tabs {
    margin-bottom: 1rem;
}
.file-upload-container {
    background-color: white;
    border: 1px solid #87cefa;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.search-container {
    background-color: white;
    border: 1px solid #87cefa;
    border-radius: 4px;
    padding: 1rem;
}
.button-primary {
    background-color: #1e90ff !important;
    color: white !important;
}
.button-primary:hover {
    background-color: #4169e1 !important;
}
"""

# 修改自定义 HTML 标题
custom_title = """
<div class="title-container">
    <div class="main-title">
        <img src="https://img.icons8.com/ios-filled/50/000000/search--v1.png" alt="Search Icon"/>
        <span>AI搜索引擎</span>
    </div>
    <div class="powered-by">Powered by 文心大模型</div>
</div>
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    gr.HTML(custom_title)

    with gr.Tabs() as tabs:
        with gr.Tab("搜索", id=0):
            with gr.Column(elem_classes="search-container"):
                query_input = gr.Textbox(
                    lines=1,
                    placeholder="想搜什么？",
                    label=None,
                    container=False,
                )

                gr.Examples(
                    examples=[
                        ["🤖人工智能的发展历史是什么？"],
                        ["🧑‍🎓2024年中国大学排名是什么？"],
                        ["🍜有哪些健康美味的食物？"]
                    ],
                    inputs=query_input
                )

                search_button = gr.Button("搜索一下", elem_classes="button-primary")
                output = gr.Markdown()

        with gr.Tab("知识库管理", id=1):
            with gr.Column(elem_classes="file-upload-container"):
                file_output = gr.Markdown()
                upload_button = gr.File(
                    file_count="multiple",
                    label="上传文本文件",
                    file_types=[".txt"]
                )
                process_button = gr.Button("生成索引", elem_classes="button-primary")

    search_button.click(
        fn=search_and_answer,
        inputs=query_input,
        outputs=output
    )

    process_button.click(
        fn=process_uploaded_files,
        inputs=upload_button,
        outputs=file_output
    )

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)