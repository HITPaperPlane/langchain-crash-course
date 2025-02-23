from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 初始化文本分割任务的pipeline
p = pipeline(task=Tasks.document_segmentation, model='damo/nlp_bert_document-segmentation_chinese-base')

# 输入需要分割的长文本
documents = '这里输入您需要分割的长文本内容，比如：这是一段示例文本，用于测试文本分割功能。希望它能正常工作。我是真有点饿了，想吃饭了'

# 执行文本分割
result = p(documents=documents)

# 更清晰地展示分割结果
print("分割后的段落：")
for i, segment in enumerate(result[OutputKeys.TEXT], 1):
    print(segment)