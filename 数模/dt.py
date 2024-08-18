import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname='/path/to/your/font.ttf')  # 例如：'/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'

# Create a directed graph
G = nx.DiGraph()

# Add nodes with hierarchical levels
G.add_node("透视检测与图像获取", level=0)
G.add_node("图像预处理", level=1)
G.add_node("去噪处理", level=2)
G.add_node("对比度增强(CLAHE)", level=2)
G.add_node("图像校准与透视变换", level=1)
G.add_node("正常场景模型构建", level=1)
G.add_node("数据库构建", level=2)
G.add_node("数据标注与整理", level=2)
G.add_node("基线模型构建(CNN, VAE)", level=2)
G.add_node("异常检测算法设计", level=1)
G.add_node("图像对比算法(差分图像, SSIM)", level=2)
G.add_node("深度学习异常检测", level=2)
G.add_node("目标物体识别与分类", level=1)
G.add_node("目标区域分割(U-Net, 多阶段分割)", level=2)
G.add_node("YOLOv8 常见危险识别", level=2)
G.add_node("模型训练与部署", level=1)
G.add_node("硬件选型", level=2)
G.add_node("数据准备与模型训练", level=2)
G.add_node("模型导出与优化", level=2)
G.add_node("实时推理引擎开发", level=2)
G.add_node("性能监控与优化", level=2)
G.add_node("系统优化与维护", level=1)
G.add_node("自动报警与人工复核", level=2)
G.add_node("持续数据更新与再训练", level=2)
G.add_node("用户界面与报告生成", level=1)
G.add_node("图像与检测结果展示", level=2)

# Add edges to connect the nodes
G.add_edges_from([
    ("透视检测与图像获取", "图像预处理"),
    ("图像预处理", "去噪处理"),
    ("图像预处理", "对比度增强(CLAHE)"),
    ("透视检测与图像获取", "图像校准与透视变换"),
    ("透视检测与图像获取", "正常场景模型构建"),
    ("正常场景模型构建", "数据库构建"),
    ("正常场景模型构建", "数据标注与整理"),
    ("正常场景模型构建", "基线模型构建(CNN, VAE)"),
    ("透视检测与图像获取", "异常检测算法设计"),
    ("异常检测算法设计", "图像对比算法(差分图像, SSIM)"),
    ("异常检测算法设计", "深度学习异常检测"),
    ("透视检测与图像获取", "目标物体识别与分类"),
    ("目标物体识别与分类", "目标区域分割(U-Net, 多阶段分割)"),
    ("目标物体识别与分类", "YOLOv8 常见危险识别"),
    ("透视检测与图像获取", "模型训练与部署"),
    ("模型训练与部署", "硬件选型"),
    ("模型训练与部署", "数据准备与模型训练"),
    ("模型训练与部署", "模型导出与优化"),
    ("模型训练与部署", "实时推理引擎开发"),
    ("模型训练与部署", "性能监控与优化"),
    ("透视检测与图像获取", "系统优化与维护"),
    ("系统优化与维护", "自动报警与人工复核"),
    ("系统优化与维护", "持续数据更新与再训练"),
    ("透视检测与图像获取", "用户界面与报告生成"),
    ("用户界面与报告生成", "图像与检测结果展示")
])

# Position the nodes in a hierarchical layout
pos = nx.multipartite_layout(G, subset_key="level")

# Draw the graph
plt.figure(figsize=(15, 10))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold", arrows=True, fontproperties=font)
plt.title("汽车底盘检测系统的实现流程思维导图", fontsize=14, fontproperties=font)
plt.show()
