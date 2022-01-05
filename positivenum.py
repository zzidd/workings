n = int(input("enter amount of numbers you want to store in the list:"))
L = []
oL = []
for i in range(n):
  num = int(input("enter any number:"))
  L.append(num)
for i in L:
  if i>0:
    oL.append(i)
print(oL)
