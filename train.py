import utils
import numpy as np

def _train(g_model, d_model, cgan_model, dataset, latent_dim, n_epochs=100, n_batch=128, save_every_epochs=10):
	'''
	g_model: generator model
	d_model: discriminator model
	cgan_model: gan_model
	dataset: dữ liệu huấn luyện, bao gồm: (X_train, y_train)
	latent_dim: Số chiều của latent space
	n_epochs: Số lượng epochs
	n_batch: Kích thước batch_size
	save_every_epochs: Số lượng epochs mà chúng ta sẽ save model.
	'''
	# Tính số lượng batch trên một epochs
	batch_per_epoch = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# Huấn luyện mô hình qua từng epochs
	for i in range(n_epochs):
		# Khởi tạo batch trên tập train
		for j in range(batch_per_epoch):
			# 1. Huấn luyện model discrinator
			# Khởi tạo batch cho ảnh real ngẫu nhiên
			[X_real, labels_real], y_real = utils._generate_real_samples(dataset, half_batch)
			# Cập nhật discriminator model weights
			d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
			# Khởi tạo batch cho ảnh fake ngẫu nhiên
			[X_fake, labels], y_fake = utils._generate_fake_samples(g_model, latent_dim, half_batch)
			# Cập nhật weights cho discriminator model
			d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
	 		# 2. Huấn luyện model generator
			# Khởi tạo các điểm ngẫu nhiên trong latent space như là đầu vào cho generator
			[z_input, labels_input] = utils._generate_latent_points(latent_dim, n_batch)
			# Khởi tạo nhãn discriminator cho các dữ liệu fake. Do chúng ta giả định là generator đánh lừa được discriminator nên nhãn của ảnh là 1.
			y_gan = np.ones((n_batch, 1))
			# Huấn luyện generator thông qua model CGAN
			g_loss = cgan_model.train_on_batch([z_input, labels_input], y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, batch_per_epoch, d_loss1, d_loss2, g_loss))
	if (i % save_every_epochs) & (i > 0):
		g_model.save('cgan_generator_epoch{}.h5'.format(i))
	# save the generator model
	g_model.save('cgan_generator.h5')