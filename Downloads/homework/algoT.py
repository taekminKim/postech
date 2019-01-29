#Python - Class
class Car :
    color = ""
    speed = 0

    def upS(self, value):
        self.speed += value

    def downS(self, value):
        self.speed -= value


mycar = Car()
