

		# # received parameters
		# self.X		  = X
		# self.X_dim    = 0
		# self.y        = y
		# self.y_error = y_error
		# self.fitted   = False
		# self.order    = order
		# self.n        = len(self.X)

		# # The method onle accepts a bool for errors if it was False
		# if type(self.y_error) == bool:
		# 	if self.y_error:
		# 		print('[Error] You must pass the values of errors')

		# 	# choose a list of ones to errors variable	
		# 	self.y_error = np.ones(len(self.X)).reshape(-1,1)

		# 	self.remove_errors = True

		# # The method onle accepts a bool for errors if it was False	
		# else:
		# 	self.remove_errors = False
			
		# print('X shape: ',self.X.shape)
		# print('y shape: ',self.y.shape)

		# # Checking if the array parameters has the same size
		# if(len(self.y)==self.n):

		# 	# corrects if the shape is not right
		# 	if(len(np.shape(self.X)) != 2 ):
		# 		print('[error] Features are not in the correct shape')
		# 	else:
		# 		self.X_dim = len(self.X[0])
		# 		self.par_dim = int(np.sum([binom(self.X_dim+k-1,k) for k in range(0,self.order+1)]))

		# 		print("ordem polinomial: ",self.order)
		# 		print("qtd. atributos  : ",self.X_dim)
		# 		print("parametr. livres: ",self.par_dim)

		# 	if(np.shape(self.y) == (self.n,1)):
		# 		self.y = np.reshape(self.y,(-1,))

		# 	if(np.shape(self.y_error) != (self.n,1)):
		# 		print('[error] Target is not in the correct shape')

		# else:
		# 	print('[error]: Number of feature vectors, targets are not the same.')

