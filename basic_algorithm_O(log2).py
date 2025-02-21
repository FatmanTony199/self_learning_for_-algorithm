Numbers = [5,17,33,41,55,61,80]
Find = 55
low = 0
high = len(Numbers) - 1

while low <= high:
    mid = (low + high) // 2
    if Numbers[mid] > Find:
        high = mid - 1
    elif Numbers[mid] < Find:
        low = mid + 1
    else:
        break
        
print(mid)