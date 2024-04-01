1. /2024-02-19-11_51_41_seplut_hdrplus-dvp_unet5_fix3d_simple_average_256_0_trans0.6-0.2-0.7_unet4：保存测试用的模型参数

2. /dataset：保存8bit和16bit的测试数据

3. structure_v3.py：提供的初始模型结构

4. structure_v6.py：将interp替换为ConvTranspose2d

   ```python 
   path = '/home/bairong.li/240106/DRC_Deploy' # 存放文件的路径
   bit = 8 # [8, 16] 选择测试图像的比特位数
   gvalue = 1.8 # 图像预处理操作，默认进行1.8的degamma
   backbone_type = 'LTM5' # [LTM4, LTM5] 选择模型架构，LTM4就是v3的结构，LTM5是将interp替换为ConvTranspose2d的结构
   ```

   

5. 







