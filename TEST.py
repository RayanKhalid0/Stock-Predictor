
def canMakeSubsequence(str1, str2):
    """
    :type str1: str
    :type str2: str
    :rtype: bool
    """
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']
    print(letters)
    arr = []
    for i in range(len(str1)):
        t = str1[i:i + len(str2)]
        arr.append(t)

    print(arr)
    for i in arr:
        if i == str2:
            return True
        print(i)

        temp = []
        for j in i:
            print(j)
            print(letters)

            x = letters.index(j) + 1
            if x == len(letters):
                x = 0
            temp.append(letters[x])

        print(temp)

    print(arr)

canMakeSubsequence("abc", "ad")