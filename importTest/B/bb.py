import sys
sys.path.append('importTest/A/')
import aa
from aa import classAA

aa.sayHello("jhh")

a = aa.classA("灿宝", id="110108")
a.sayHello()
a.birthday()



a2=classAA()

a2.sayHello()
a2.birthday()
#print(globalData)