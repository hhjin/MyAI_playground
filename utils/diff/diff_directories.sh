#!/bin/bash

 
echo "这个程序可以比较两个目录下(包含所有子目录)的文件差异"
echo "Usage :  ./diff_directories.sh  dir1  dir2 " 
if [ $# -ne 2 ]; then
  echo "请输入两个参数，第一个参数为目录1，第二个参数为目录2"
  exit 1
fi

path1="$1"
path2="$2"

get_subdirs() {
  local path="$1"
  find "$path" -type d -printf '%P\n'
}

diff_directories() {
  local path1="$1"
  local path2="$2"

  subdirs1=$(get_subdirs "$path1")
  subdirs2=$(get_subdirs "$path2")

  common_subdirs=$(comm -12 <(echo "$subdirs1" | sort) <(echo "$subdirs2" | sort))

  for subdir in $common_subdirs; do
    dir1="$path1/$subdir"
    dir2="$path2/$subdir"
 
    echo " "
    echo  "比较目录 $dir1 和 $dir2"
    diff -r "$dir1" "$dir2"
  done
 
 
#echo "path1 和 path1 不同子目录如下："
#diff <(echo "$subdirs1") <(echo "$subdirs2") | grep -E "^<|^>" | sed 's/< /目录1中不存在的项目: /g;s/> /目录2中不存在的项目: /g'

}

diff_directories "$path1" "$path2"