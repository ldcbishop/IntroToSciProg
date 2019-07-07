from colorama import Fore, Back, Style
import numpy as np

class Tester:
	def testQuestion1(self,ui1):
		is_correct = False
		if ui1 == [5, 'yes', 3.4]:
			answer = True
		else:
			answer = False
		self.isItCorrect(answer)

	def testQuestion2(self,ui2):
		numbers = np.arange(0,5000,3)
		this_sum = 0
		for i in numbers:
			this_sum += i
		test = (ui2 == this_sum)
		self.isItCorrect(test)

	def testQuestion3(self,ui3):
		numbers = np.arange(0,5000,3)
		this_sum = 0
		for i in numbers:
			if i % 2 == 1:
				this_sum += i
		self.isItCorrect(ui3==this_sum)

	def testQuestion4(self,ui4_1,ui4_2):
		numbers = np.arange(0,5000,3)
		this_sum = 0 
		that_sum = 0
		for i in numbers:
			if i % 2 == 1:
				this_sum += i
			elif int(i**0.5 + 0.5)**2 == i:
				that_sum += i
		test = ui4_1 == this_sum and ui4_2 == that_sum
		self.isItCorrect(test)

	def testQuestion5(self, ui5):
		numbers = np.arange(0,5000,3)
		odds = 0
		evens = 0
		for i in numbers:
			if i % 2 == 0:
				evens+= i
			else:
				odds+= i
		final_val = np.round(float(evens)/float(odds),2)
		self.isItCorrect(final_val == ui5)

	def testQuestion6(self,ui6):
		numbers = np.arange(0,5000,3)
		self.testQuestion5(ui6(numbers))

	def testQuestion7(self,ui7):
		numbers = np.arange(0,5000,3)
		self.testQuestion5(ui7.evenAndodds(numbers))

	def testQuestion8(self,ui6):
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

# This is the class I wrote to satisfy question 3
