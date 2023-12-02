class CARS:
    def __init__(self, name, brand,engine, milage, top_speed):
        self.name=name
        self.brand=brand
        self.engine = engine
        self.millage=milage
        self.top_speed=top_speed
    
    def carinfo(self):
        print(f"This{self.brand} {self.name} has an Engine size of {self.engine} it's current mileage is {self.millage} with top speed {self.top_speed}")



bmw1=CARS("BMW","Bmw", "V6", 20,"350")
bmw1.carinfo()
# Example 1: Audi A6
audi_a6 = CARS("Audi", "A6", "2.0 TFSI", 20, "350")
audi_a6.carinfo()
# Example 2: Mercedes-Benz S-Class
mercedes_s_class = CARS("Mercedes-Benz", "S-Class", "3.0 Biturbo", 20, "450")
mercedes_s_class.carinfo()
# Example 3: Lexus RX
lexus_rx = CARS("Lexus", "RX", "3.5 V6", 20, "350")
lexus_rx.carinfo()
# Example 4: Toyota Camry
toyota_camry = CARS("Toyota", "Camry", "2.5 GT", 20, "300")
toyota_camry.carinfo()
# Example 5: Ford Mustang
ford_mustang = CARS("Ford", "Mustang", "5.0 Coyote", 20, "450")
ford_mustang.carinfo()

print("hello")