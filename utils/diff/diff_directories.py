import os
import subprocess

def get_subdirs(path):
    subdirs = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            subdir = os.path.relpath(os.path.join(root, dir), path)
            subdirs.append(subdir)
    return subdirs

def diff_directories(path1, path2):
    subdirs1 = get_subdirs(path1)
    subdirs2 = get_subdirs(path2)

    common_subdirs = list(set(subdirs1) & set(subdirs2))

    for subdir in common_subdirs:
        dir1 = os.path.join(path1, subdir)
        dir2 = os.path.join(path2, subdir)
        result = subprocess.run(["diff", "-r", dir1, dir2], capture_output=True, text=True)
        print(result.stdout)

if __name__ == "__main__":
    path1 = input("请输入第一个目录名：")
    path2 = input("请输入第二个目录名：")
    diff_directories(path1, path2)