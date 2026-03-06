def remove_first_char(file_path):
  """
  删除文件每行第一个字符
  """
  with open(file_path, 'r') as f:
    lines = f.readlines()

  with open(file_path, 'w') as f:
    for line in lines:
      if line.startswith("0/"):
        # 删除每行第一个字符
        f.write(line[1:])
      else:
        f.write(line)

# 应用函数
remove_first_char("our_yolov7_tiny_result.txt")
remove_first_char("yolov7_tiny_result.txt")
