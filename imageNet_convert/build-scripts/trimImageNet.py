#run from bigisland

import os

f = open("/home/yli/nvme_ssd/imagenet/build-scripts/imagenet_lsvrc_2015_synsets.txt", "r")
tars = "/home/yli/nvme_ssd/imagenet/ilsvrc2012/data/ILSVRC2012_img_train-bak/tars-bak/"

train_dir = "/home/yli/nvme_ssd/imagenet/ilsvrc2012/data/train/"
#os.system("cd /home/yli/nvme_ssd/imagenet/ilsvrc2012/data/train")

for line in f.readlines():
#  print(line) 
  line = line.strip('\n')
  img_path = train_dir + line
  os.system("mkdir " + img_path)
  os.system("cp " + tars + line + ".tar  " + img_path)
  os.system("cd " + img_path + " && tar xvf *.tar > /dev/null && rm -f *.tar")
