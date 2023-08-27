from typing import Any, Dict, List, Optional

def sayHello(name):
    print ('hello '+name)

class classA():
 
    age : Optional[str]="3"

    def __init__(self,name, id):
        self.name = name
        self.id = id

    def sayHello(self)    :
         sayHello(self.name+"  ID:"+self.id)
    
    def birthday(self)    :
        print (self.name+" , Happy to your "+self.age+" years Birthday!" )


class classAA():
 
    age : Optional[str]="14"
    name: Optional[str]="Angle"
    id :  Optional[str]="431229"


    def sayHello(self)    :
         sayHello(self.name+"  ID:"+self.id)
    
    def birthday(self)    :
        print (self.name+" , Happy to your "+self.age+" years Birthday!" )