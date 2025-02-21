Numbers = [41,33,17,80,61,5,55]

length = len(Numbers)

for i in range(1, length):
    position = i
    value = Numbers[i]

    while position > 0 and Numbers[position-1] > Numbers[position]:
        Numbers[position], Numbers[position-1] = Numbers[position-1], Numbers[position]
        position -= 1

