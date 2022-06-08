def ave (n):
    return sum(n) / len(n)


NOofstudents = int(input("Enter the number of  sudents : "))
studentdic = {}
grades =[]

print("Enter the name first the enter his 3 grades and so on \n")
for i  in range(NOofstudents):
    name = input("Enter the name : ").lower()
    grades = []
    for k in range(3):
        grades.append(int(input()))
    studentdic[name] = grades
try:
    chosen_st = input("enter the student you want : ").lower()
    average = ave(studentdic[chosen_st])

    print("the average of marks obtained by {} is {} ".format(chosen_st , average))

except:
    print("There is no student with this name")