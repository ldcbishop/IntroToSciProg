from colorama import Fore, Back, Style

class Tester:

	def testQuestion1(self,ui1):

		is_correct = False
		if ui1 == [5, 'yes', 2.0]:
			answer = True
		else:
			answer = False
		self.isItCorrect(answer)

	def testQuestion2(self,ui2):
		my_list = [2,3,4,5,8,7,16,11,32,13,64,17,32,19,16,23,8,29,4,31,2]
		sum_ev = 0.0
		sum_odd = 0.0
		for val in my_list:
			if val % 2 == 0:
				sum_ev += val
			else:
				sum_odd += val
		test = ui2(my_list) == sum_ev/sum_odd
		self.isItCorrect(test)

	def testQuestion3(self,ui3):
		x = ui3.x
		y = ui3.y
		z = ui3.z
		answer = True
		if ui3.AreaOfCircle() == 3.14*(x-y)**2:
			answer &= True
		else:
			print(Fore.RED+'The calculation in AreaOfCircle() is incorrect!')

		if ui3.VolumeOfCone() == ui3.AreaOfCircle() * z/3:
			answer &= True
		else:
			print(Fore.RED+'The calculation in VolumeOfCone() is incorrect!')

		self.isItCorrect(answer)

	def isItCorrect(self, answer):
		if answer:
			print(Fore.GREEN+'That is correct!')
		else:
			print(Fore.RED+'Please try again!')
