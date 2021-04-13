import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def draw_plot(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        nums = f.readlines()
    nums = [float(n.strip()) for i,n in enumerate(nums)]
    nums = nums[404750:]
    x = list(range(0,len(nums)))
    plt.figure(figsize=(15, 10))
    plt.ylabel("Loss", fontsize=20)
    plt.xlabel("step", fontsize=20)
    plt.xticks(fontsize=20, family='Times New Roman')
    plt.yticks(fontsize=20, family='Times New Roman')
    plt.title("RNN损失函数", fontsize=30)
    plt.plot(x, nums, c="r")
    plt.savefig('RNN损失函数.jpg')
    plt.show()

if __name__ == "__main__":
    file_path = "./log/rnnloss_log.txt"
    draw_plot(file_path)