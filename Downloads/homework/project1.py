f = open("students.txt",'r')
data = list()
# 이중리스트를 이용하여 데이터 관리
for line in f:
    temp = line.split()
    data.append(temp)

f.close()
for i in range(len(data)):
    data[i][3] = int(data[i][3])
    data[i][4] = int(data[i][4])

def calAG(a, b):
    avg = ( a + b )/ 2
    if avg >= 90:
        G = "A"
    elif avg >= 80 and avg < 90:
        G = "B"
    elif avg >= 70 and avg < 80:
        G = "C"
    elif avg >= 60 and avg < 70:
        G = "D"
    else:
        G = "F"
    return avg, G
def printHeader():
    print("%2s\t%16s\t%3s\t%2s\t%s\t%2s" % (
        "Student", "Name", "Midterm", "Final", "Average", "Grade" ))
    print("----------------------------------------------------------")
def printInfoRow(i):
    print("%s\t%12s\t%4d\t%4d\t%2.1f\t%2s" % (
    data[i][0], (data[i][1] + " " + data[i][2]), data[i][3], data[i][4], data[i][5], data[i][6]))

#평균과 학점은 이중리스트에서 데이터를 추출하여 리스트에 append 한다.
for i in range(len(data)):
    average,grade = calAG(data[i][3], data[i][4])
    data[i].append(average)
    data[i].append(grade)

#0:학번 1:성 2:이름 3:중간점수 4:기말점수 5:평균 6:학점
def showInfo(data):
    data.sort(key=lambda e: e[5])
    data.reverse()
    printHeader()
    for i in range(len(data)):
        printInfoRow(i)
def searchStu(data):
    id = input("Student ID: ")
    flag = 0
    for i in range(len(data)):
        if id == data[i][0]:
            flag = 1
            printHeader()
            printInfoRow(i)
    if flag == 0:
        print("NO SUCH PERSON.\n")
def changeScore(data):
    id = input("Student ID: ")
    flag = 0
    for i in range(len(data)):
        if id == data[i][0]:
            flag = 1
            op1 = input("Mid/Final? ")
            if op1 == "mid" or op1 == "final":
                op2 = int(input("Input new score: "))
                if op2 >= 0 and op2 <= 100:
                    printHeader()
                    # 모든 조건을 만족하므로 점수를 수정한다.
                    printInfoRow(i)
                    print("Score changed.")
                    if op1 == "mid":
                        newMid = op2
                        # 5 avg 6 gr
                        average, grade = calAG(newMid, int(data[i][4]))
                        data[i][3] = newMid
                        data[i][5] = average
                        data[i][6] = grade
                        printInfoRow(i)
                    elif op1 == "final":
                        newFinal = op2
                        average, grade = calAG(int(data[i][3]), newFinal)
                        data[i][4] = newFinal
                        data[i][5] = average
                        data[i][6] = grade
                        printInfoRow(i)
                    break
                else:
                    print()
                    continue
            else:
                print()
                continue
    if flag == 0:
        print("NO SUCH PERSON.\n")
def addStu(data):
    id = input("Student ID: ")
    flag = 0
    for i in range(len(data)):
        if id == data[i][0]:
            print("ALREADY EXISTS.\n")
            flag = 1
    if flag == 0:
        name = input("Name: ")
        nameList = name.split()
        midScore = int(input("Midterm Score: "))
        finalScore = int(input("Final Score: "))
        average, grade = calAG(midScore, finalScore)
        data.append([id, nameList[0], nameList[1], midScore, finalScore, average, grade])
        print("Student added.\n")
def searchGr(data):
    grade = input("Grade to search: ")
    flag = 0
    header = 0
    for i in range(len(data)):
        if grade not in "ABCDF":
            print()
            flag = 2
            break
        elif grade == data[i][6]:
            flag = 1
            if header == 0:
                printHeader()
                printInfoRow(i)
                header = 1
            else:
                printInfoRow(i)
    if flag == 0:
        print("NO RESULTS.\n")
def removeInfo(data):
    if len(data) == 0:
        print("List is empty.\n")
    else:
        id = input("Student ID: ")
        flag = 0
        for i in range(len(data)):
            if id in data[i]:
                # 학번을 찾았으므로 삭제처리 해야함
                flag = 1
                del data[i]
                print("Student removed.\n")
                break
        if flag == 0:
            print("NO SUCH PERSON.\n")
def quit(data):
    ask = input("Save data?[yes/no] ")
    if ask == "yes":
        fName = input("File name: ")
        data.sort(key=lambda e: e[5])
        data.reverse()
        f = open(fName, "w")
        writeBuffer = ""
        for i in range(len(data)):
            writeBuffer += data[i][0] + "\t" + (data[i][1] + " " + data[i][2]) + "\t" + str(data[i][3]) + "\t" + str(
                data[i][4]) + "\n"
        f.write(writeBuffer)
        f.close()
        exit()
    else:
        print()

while True:
    cm = input("#")
    command = cm.upper()
    #show, search, changescore, searchgrade, add, remove, quit
    if command == "SHOW":
        showInfo(data)
    elif command == "SEARCH":
        searchStu(data)
    elif command == "CHANGESCORE":
        changeScore(data)
    elif command == "ADD":
        addStu(data)
    elif command == "SEARCHGRADE":
        searchGr(data)
    elif command == "REMOVE":
        removeInfo(data)
    elif command == "QUIT":
        quit(data)
