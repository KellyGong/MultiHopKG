x = input()

res = []

while x != 'z':
    x_list = x.split('=')
    if len(x_list) == 2:
        x_list[1] = x_list[1].replace('"', '')
        x_list[0] = x_list[0].replace('"', '')
        res.append(x_list)
    x = input()
    
for i in res:
    print(f'"--{i[0]}", "{i[1]}",')
