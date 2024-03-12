import random
class filterhandler:
	def filter_handler_latent_x(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		if self.latent[0] == args[0]:
			return
		
		self.latent[0] = args[0]
		self.mustRun = True
		return

	def filter_handler_latent_y(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		if self.latent[1] == args[0]:
			return
		
		self.latent[1] = args[0]
		self.mustRun = True
		return

	#-1 - 2 default 1
	def filter_handler_truncpsi(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['trunc_psi'] == args[0]:
			return
		
		self.renderArgs['trunc_psi'] = args[0]
		self.mustRun = True

	#input transform x0
	def filter_handler_transformx0(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][0][0] == args[0]:
			return
		
		self.renderArgs['input_transform'][0][0] = args[0]
		self.mustRun = True

	#input transform y0
	def filter_handler_transformy0(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][0][1] == args[0]:
			return
		
		self.renderArgs['input_transform'][0][1] = args[0]
		self.mustRun = True

	#input transform z0
	def filter_handler_transformz0(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][0][2] == args[0]:
			return
		
		self.renderArgs['input_transform'][0][2] = args[0]
		self.mustRun = True

	#input transform x1
	def filter_handler_transformx1(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][1][0] == args[0]:
			return
		
		self.renderArgs['input_transform'][1][0] = args[0]
		self.mustRun = True

	#input transform y1
	def filter_handler_transformy1(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][1][1] == args[0]:
			return
		
		self.renderArgs['input_transform'][1][1] = args[0]
		self.mustRun = True

	#input transform z1
	def filter_handler_transformz1(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][1][2] == args[0]:
			return
		
		self.renderArgs['input_transform'][1][2] = args[0]
		self.mustRun = True

	#input transform x2
	def filter_handler_transformx2(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][2][0] == args[0]:
			return
		
		self.renderArgs['input_transform'][2][0] = args[0]
		self.mustRun = True

	#input transform y2
	def filter_handler_transformy2(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][2][1] == args[0]:
			return
		
		self.renderArgs['input_transform'][2][1] = args[0]
		self.mustRun = True

	#input transform z2
	def filter_handler_transformz2(self, address, *args):
		# We expect one float argument
		if not len(args) == 1 or type(args[0]) is not float:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['input_transform'][2][2] == args[0]:
			return
		
		self.renderArgs['input_transform'][2][2] = args[0]
		self.mustRun = True

	# 0- 16 default 16
	def filter_handler_trunccutoff(self, address, *args):
		# We expect one int argument
		if not len(args) == 1 or type(args[0]) is not int:
			return
		
		# do nothing if latens are the same
		if self.renderArgs['trunc_cutoff'] == args[0]:
			return
		
		self.renderArgs['trunc_cutoff'] = args[0]
		self.mustRun = True

	def filter_handler_randomize(self, address, *args):
		self.randFactor[0] = random.randint(0,1611312)
		self.randFactor[1] = random.randint(0,1611312)
		self.mustRun = True

	def filter_handler_targetfps(self, address, *args):
		if not len(args) == 1 or type(args[0]) is not int:
			return
		self.fps = args[0]

	def filter_handler_setpkl(self, address, *args):
		if not len(args) == 1 or type(args[0]) is not str:
			return
		
		if (not os.path.isfile(args[0])):
			print ("Could not load pkl file "+args[0]+" - file does not exist")
			return

		print('loading pklfile '+args[0], end='\r')
		self.renderArgs.pkl = args[0]
		self.modelChanged = True