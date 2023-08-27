import opencc

# 定义输入和输出文件的路径
traditional_file = "~/whisper.cpp/WAVs/一个AI创业者_whisperccp_medium.md"
simplified_file =  "~/whisper.cpp/WAVs/一个AI创业者_whisperccp_mediumS.md"

# 创建OpenCC转换器实例
converter = opencc.OpenCC("t2s.json")

# 打开繁体中文文件并读取内容
with open(traditional_file, "r") as file:
    traditional_text = file.read()

# 使用OpenCC将繁体中文转换为简体中文
simplified_text = converter.convert(traditional_text)

# 打开简体中文文件并写入转换后的内容
with open(simplified_file, "w") as file:
    file.write(simplified_text)