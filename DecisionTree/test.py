group = ["a","a","b","b","a","c","b","c","a","a","a","a","a","a","a","a","a","a","a","a","a","a"]
sum_dict = dict()
for x in group:
    try:
        sum_dict[x]+=1
    except:
        sum_dict[x]=1

for x in sum_dict.values():
    if float(100/(len(group)/x)) > 75.0:
        print("Make terminal node")

