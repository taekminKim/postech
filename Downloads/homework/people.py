
class Person:
    name =""
    age = 0
    department = ""
    def __init__(self, name, age, department):
        self.name = name
        self.age = age
        self.department = department

    def get_name(self):
        return self.name


class Student(Person):
    id=0
    GPA =0.0
    advisor=""
    Professor = None
    def __init__(self, name, age, department, id, GPA):
        Person.__init__(self, name=name, age=age, department= department)
        self.id = id
        self.GPA = GPA

    def print_info(self):
        print("'제 이름은 %s, 나이는 %d, 학과는 %s, 지도교수님은 %s 입니다'"
              %(self.name, self.age, self.department, self.advisor))

    def reg_advisor(self, Professor):
        self.advisor = Professor.get_name()


class Professor(Person):
    position=""
    laboratory=""
    student = []
    def __init__(self, name, age, department, position, laboratory):
        Person.__init__(self, name=name, age=age, department= department)
        self.position = position
        self.laboratory = laboratory

    def reg_student(self, Student):
        self.student.append(Student.name)

    def print_info(self):
        print("'제 이름은 %s, 나이는 %d, 학과는 %s, 지도학생은 %s 입니다'"
              %(self.name, self.age, self.department, ",".join(self.student)))

stu1 = Student('Kim', 30, 'Computer',20001234, 4.5)
stu2 = Student('Lee', 22, 'Computer', 20101234, 0.5)
prof1 = Professor('Lee', 55, 'Computer', 'Full', 'KLE')


stu1.reg_advisor(prof1)
stu2.reg_advisor(prof1)
prof1.reg_student(stu1)
prof1.reg_student(stu2)

stu1.print_info()
stu2.print_info()
prof1.print_info()