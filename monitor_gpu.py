'''
Author: lusz
Date: 2024-12-04 22:30:17
Description: 
'''
import time
import subprocess

def monitor_gpu_to_file(output_file="gpu_monitor/products_nccl.txt", interval=0.01):
    """
    监控 GPU 利用率并保存到文件。

    Args:
        output_file (str): 保存的文件名。
        interval (float): 每次采样的间隔时间，单位秒。
    """
    duration = 0.2  # 固定监控时长为 20 秒
    start_time = time.time()
    with open(output_file, "w") as file:
        try:
            while time.time() - start_time < duration:
                # 获取 GPU 利用率
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
                ).decode("utf-8")
                utilization = output.strip().split("\n")
                
                # 将数据以空格分隔写入文件
                file.write(" ".join(utilization) + "\n")
                file.flush()  # 确保实时写入
                time.sleep(interval)
        except KeyboardInterrupt:
            print("监控已手动终止。")
        except FileNotFoundError:
            print("未找到 nvidia-smi，请确保 NVIDIA 驱动已安装。")

if __name__ == "__main__":
    # 保存文件名为 `gpu_monitor/products_nccl.txt`，监控时间为 20 秒，每 0.01 秒记录一次
    monitor_gpu_to_file(output_file="gpu_monitor/reddit_.txt", interval=0.01)
