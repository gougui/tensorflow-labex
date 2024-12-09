# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import re
import os
#model_dir='/root/test/'

# 添加日志输出
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('./'))

model_dir='./'
#将类别ID转换为人类易读的标签
class NodeLookup(object):
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    
    # 检查文件是否存在
    print("Checking if label_lookup_path exists:", os.path.exists(label_lookup_path))
    print("Checking if uid_lookup_path exists:", os.path.exists(uid_lookup_path))
    
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.io.gfile.exists(uid_lookup_path):
      tf.compat.v1.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.io.gfile.exists(label_lookup_path):
      tf.compat.v1.logging.fatal('File does not exist %s', label_lookup_path)
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.io.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string
    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.io.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.compat.v1.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
    return node_id_to_name
  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]
#读取训练好的Inception-v3模型来创建graph
def create_graph():
  with tf.io.gfile.GFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def classify_graph(imageFile):

  # 检查图片文件是否存在
  if not os.path.exists(imageFile):
        print(f"Error: Image file not found: {imageFile}")
        return
        
  print(f"Processing image: {imageFile}")

  #读取图片
  image_data = tf.io.gfile.GFile(imageFile, 'rb').read()
  #创建graph
  create_graph()
  sess=tf.compat.v1.Session()
  #Inception-v3模型的最后一层softmax的输出
  softmax_tensor= sess.graph.get_tensor_by_name('softmax:0')
  #输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
  predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
  #(1,1008)->(1008,)
  predictions = np.squeeze(predictions)
  # ID --> English string label.
  node_lookup = NodeLookup()
  #取出前5个概率最大的值（top-5)
  top_5 = predictions.argsort()[-5:][::-1]
  for node_id in top_5:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))
  sess.close()

if __name__ == "__main__":
  imageDir = "/root/data/images"
  # imageDir = "/root/code/tensorflow-labex/images"

    # 检查图片目录是否存在
  if not os.path.exists(imageDir):
        print(f"Error: Image directory not found: {imageDir}")
        exit(1)
        
  print(f"Scanning directory: {imageDir}")
  # print(f"Files in /root/data: {os.listdir('/root/data/images')}")
  print(f"Files in image directory: {os.listdir(imageDir)}")
  

  for root, dirs, files in os.walk(imageDir):
    for f in files:
      print(os.path.join(root, f))
      classify_graph(os.path.join(root, f))
      print("\n")
    
