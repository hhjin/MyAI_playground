def reverse_stringGLM(s):
   # Create a stack to hold the characters of the string
   stack = []
   # Go through the characters in the string and push them onto the stack
   for char in s:
       stack.append(char)
   # Return the reversed string
   return stack.pop()

def reverse_stringGLM2(s):
   # Initialize variables to keep track of characters and their indices
   chars = []
   i = 0
   for char in s:
       # Add the character to the list of characters
       chars.append(char)
       # Increment the index of the character
       i += 1
   
   # Reverse the order of the characters
   for i in range(len(chars) - 1, -1, -1):
       chars[i], chars[i + 1] = chars[i + 1], chars[i]
   
   # Return the reversed string
   return ''.join(chars)

def reverse_stringYunFei(s):
    return s[::-1]


def reverse_stringAzure(string):
  reversed_string = ""
  for i in range(len(string)-1, -1, -1):
    reversed_string += string[i]
  return reversed_string

print(reverse_stringGLM("123456"))